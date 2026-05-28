/**
 * CalibrationChart — Roadmap 0.3
 *
 * Reliability diagram: predicted probability (bins) vs observed frequency.
 * With confidence band per bin and histogram of prediction counts.
 *
 * "When we say 30%, it happens ~30%" — THE sales argument.
 * Investor-readable: well-calibrated = dots on the diagonal.
 */
import { useRef, useEffect } from 'react';
import * as d3 from 'd3';

export interface CalibrationBin {
  bin_lo: number;
  bin_hi: number;
  mean_predicted: number;
  observed_freq: number | null;
  count: number;
}

interface CalibrationChartProps {
  bins: CalibrationBin[];
  ece?: number | null;
  nGames?: number;
  source?: string;
}

export default function CalibrationChart({ bins, ece, nGames, source }: CalibrationChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !bins.length) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const W = 340, H = 280;
    const margin = { top: 20, right: 20, bottom: 50, left: 50 };
    const innerW = W - margin.left - margin.right;
    const innerH = H - margin.top - margin.bottom;
    const histH = 32; // histogram strip height
    const mainH = innerH - histH - 8;

    svg.attr('width', W).attr('height', H);

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear().domain([0, 1]).range([0, innerW]);
    const yScale = d3.scaleLinear().domain([0, 1]).range([mainH, 0]);

    // ── Background grid ──────────────────────────────────────────────────
    [0.2, 0.4, 0.6, 0.8].forEach(v => {
      g.append('line')
        .attr('x1', 0).attr('x2', innerW)
        .attr('y1', yScale(v)).attr('y2', yScale(v))
        .attr('stroke', 'rgba(255,255,255,0.05)').attr('stroke-width', 1);
      g.append('line')
        .attr('x1', xScale(v)).attr('x2', xScale(v))
        .attr('y1', 0).attr('y2', mainH)
        .attr('stroke', 'rgba(255,255,255,0.05)').attr('stroke-width', 1);
    });

    // ── Perfect calibration diagonal ─────────────────────────────────────
    g.append('line')
      .attr('x1', 0).attr('y1', mainH)
      .attr('x2', innerW).attr('y2', 0)
      .attr('stroke', 'rgba(255,255,255,0.35)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '5,3');

    g.append('text')
      .attr('x', innerW - 4)
      .attr('y', 4)
      .attr('fill', 'rgba(255,255,255,0.3)')
      .attr('font-size', '9px')
      .attr('text-anchor', 'end')
      .attr('font-family', 'JetBrains Mono, monospace')
      .text('Calibración perfecta');

    // ── ±0.1 confidence band around diagonal ─────────────────────────────
    const bandArea = d3.area<[number, number]>()
      .x(d => xScale(d[0]))
      .y0(d => yScale(Math.max(0, d[0] - 0.1)))
      .y1(d => yScale(Math.min(1, d[0] + 0.1)));

    const diagonalPoints: [number, number][] = [[0, 0], [1, 1]];
    g.append('path')
      .datum(diagonalPoints)
      .attr('fill', 'rgba(255,255,255,0.04)')
      .attr('d', bandArea);

    // ── Calibration dots ─────────────────────────────────────────────────
    const validBins = bins.filter(b => b.observed_freq !== null && b.count > 0);

    // Connecting line
    if (validBins.length > 1) {
      const line = d3.line<CalibrationBin>()
        .x(b => xScale(b.mean_predicted))
        .y(b => yScale(b.observed_freq!))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(validBins)
        .attr('fill', 'none')
        .attr('stroke', '#3B82F6')
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.6)
        .attr('d', line);
    }

    // Dots
    validBins.forEach(b => {
      const overConf = b.observed_freq! > b.mean_predicted + 0.05;
      const underConf = b.observed_freq! < b.mean_predicted - 0.05;
      const color = overConf ? '#22C55E' : underConf ? '#EF4444' : '#60A5FA';

      // Error line (deviation from perfect)
      g.append('line')
        .attr('x1', xScale(b.mean_predicted))
        .attr('y1', yScale(b.mean_predicted))
        .attr('x2', xScale(b.mean_predicted))
        .attr('y2', yScale(b.observed_freq!))
        .attr('stroke', color)
        .attr('stroke-width', 1.5)
        .attr('opacity', 0.6);

      g.append('circle')
        .attr('cx', xScale(b.mean_predicted))
        .attr('cy', yScale(b.observed_freq!))
        .attr('r', 5)
        .attr('fill', color)
        .attr('stroke', '#0A0E1A')
        .attr('stroke-width', 1.5);
    });

    // ── Axes ─────────────────────────────────────────────────────────────
    const xAxis = g.append('g').attr('transform', `translate(0,${mainH})`);
    xAxis.append('line').attr('x2', innerW).attr('stroke', 'rgba(255,255,255,0.15)');
    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      xAxis.append('text')
        .attr('x', xScale(v)).attr('y', 14)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.3)')
        .attr('font-size', '9px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(`${(v * 100).toFixed(0)}%`);
    });

    const yAxis = g.append('g');
    yAxis.append('line').attr('y2', mainH).attr('stroke', 'rgba(255,255,255,0.15)');
    [0, 0.2, 0.4, 0.6, 0.8, 1.0].forEach(v => {
      yAxis.append('text')
        .attr('x', -6).attr('y', yScale(v))
        .attr('text-anchor', 'end')
        .attr('dominant-baseline', 'middle')
        .attr('fill', 'rgba(255,255,255,0.3)')
        .attr('font-size', '9px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(`${(v * 100).toFixed(0)}%`);
    });

    // Axis labels
    g.append('text')
      .attr('x', innerW / 2).attr('y', mainH + 40)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.25)')
      .attr('font-size', '9px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .text('Probabilidad predicha');

    g.append('text')
      .attr('transform', `rotate(-90)`)
      .attr('x', -mainH / 2).attr('y', -38)
      .attr('text-anchor', 'middle')
      .attr('fill', 'rgba(255,255,255,0.25)')
      .attr('font-size', '9px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .text('Frecuencia observada');

    // ── Count histogram strip ─────────────────────────────────────────────
    const histY = mainH + 8;
    const maxCount = Math.max(1, d3.max(bins, b => b.count) ?? 1);
    const histScale = d3.scaleLinear().domain([0, maxCount]).range([0, histH - 4]);
    const binW = innerW / bins.length;

    bins.forEach((b, i) => {
      if (!b.count) return;
      const hh = histScale(b.count);
      g.append('rect')
        .attr('x', i * binW + 1)
        .attr('y', histY + (histH - hh) - 4)
        .attr('width', binW - 2)
        .attr('height', hh)
        .attr('fill', '#1E3A5F')
        .attr('rx', 1);

      if (b.count >= 3) {
        g.append('text')
          .attr('x', i * binW + binW / 2)
          .attr('y', histY + histH - 6)
          .attr('text-anchor', 'middle')
          .attr('fill', 'rgba(255,255,255,0.3)')
          .attr('font-size', '8px')
          .attr('font-family', 'JetBrains Mono, monospace')
          .text(b.count);
      }
    });

  }, [bins]);

  const eceColor = ece !== null && ece !== undefined
    ? ece < 0.05 ? '#22C55E' : ece < 0.10 ? '#FFB800' : '#EF4444'
    : '#6B7280';
  const eceLabel = ece !== null && ece !== undefined
    ? ece < 0.05 ? 'Bien calibrado' : ece < 0.10 ? 'Aceptable' : 'Descalibrado'
    : '—';

  return (
    <div>
      {/* Hero number: ECE */}
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 10, marginBottom: 10 }}>
        <div>
          <p style={{ fontSize: '0.60rem', color: '#374151', letterSpacing: '0.12em',
                      textTransform: 'uppercase', marginBottom: 2 }}>
            ECE — Error de Calibración Esperado
          </p>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
            <span style={{
              fontSize: '1.8rem', fontWeight: 800, fontFamily: 'JetBrains Mono, monospace',
              color: eceColor,
            }}>
              {ece !== null && ece !== undefined ? ece.toFixed(4) : '—'}
            </span>
            <span style={{ fontSize: '0.72rem', color: eceColor, fontWeight: 600 }}>
              {eceLabel}
            </span>
          </div>
          <p style={{ fontSize: '0.65rem', color: '#374151', marginTop: 2 }}>
            n = {nGames ?? 0} predicciones
            {source === 'demo' && ' · datos de demostración'}
            {source === 'backtest_file' && ' · backtest.py'}
          </p>
        </div>
        <div style={{ marginLeft: 'auto', fontSize: '0.68rem', color: '#374151',
                      fontFamily: 'JetBrains Mono, monospace', textAlign: 'right', lineHeight: 1.7 }}>
          <p>● Verde = observado {'>'} predicho</p>
          <p>● Rojo = observado {'<'} predicho</p>
          <p>● Azul = bien calibrado</p>
        </div>
      </div>

      <div style={{ overflowX: 'auto' }}>
        <svg ref={svgRef} width={340} height={280} />
      </div>

      <p style={{ fontSize: '0.62rem', color: '#1F2937', marginTop: 6,
                  fontFamily: 'JetBrains Mono, monospace' }}>
        ECE = 0 → calibración perfecta · Barras grises = n predicciones por bin
      </p>
    </div>
  );
}
