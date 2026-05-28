/**
 * ProjectionChart — Box-and-whisker (Bug 2 fix).
 *
 * Replaces the plain bar with a full distributional view:
 *  • Whiskers at P10 / P90
 *  • IQR box (P25 – P75) with uncertainty colour-coding
 *  • Median marker
 *  • E[R] mean label
 *  • Actual-runs marker (post-game overlay)
 *  • σ badge with WCAG-AA colour + shape encoding
 *
 * Built with D3 — Recharts cannot render the required overlapping geometry.
 */
import { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface ProjectionChartProps {
  /** Expected runs (mean). */
  er: number;
  std_dev?: number;
  percentile_10?: number;
  percentile_25?: number;
  percentile_75?: number;
  percentile_90?: number;
  uncertainty_level?: 'low' | 'medium' | 'high';
  /** Actual runs scored (post-game overlay). Omit for pre-game. */
  actual_runs?: number;
  /** Simulation count for tooltip annotation. */
  simulation_n?: number;
  className?: string;
}

// WCAG AA compliant palette — colour + symbol always used together
const UNCERTAINTY_COLOR: Record<string, string> = {
  low:    '#22C55E',  // green
  medium: '#F59E0B',  // amber
  high:   '#EF4444',  // red
};
const UNCERTAINTY_LABEL: Record<string, string> = {
  low:    'Proyección confiable (σ < 0.8)',
  medium: 'Incertidumbre moderada (σ 0.8–1.5)',
  high:   'Alta varianza — decisión de riesgo (σ > 1.5)',
};
const UNCERTAINTY_SYMBOL: Record<string, string> = {
  low: '●', medium: '◆', high: '▲',
};

export default function ProjectionChart({
  er,
  std_dev,
  percentile_10,
  percentile_25,
  percentile_75,
  percentile_90,
  uncertainty_level = 'medium',
  actual_runs,
  simulation_n,
  className = '',
}: ProjectionChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const uColor = UNCERTAINTY_COLOR[uncertainty_level] ?? '#F59E0B';

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width   = containerRef.current.clientWidth || 340;
    const height  = 155;   // taller to fit KDE curve above the box
    const margin  = { top: 32, right: 20, bottom: 32, left: 20 };
    const innerW  = width - margin.left - margin.right;
    const innerH  = height - margin.top - margin.bottom;
    const midY    = innerH * 0.62;  // shift center down slightly for KDE room
    const boxH    = innerH * 0.35;

    const sigma = std_dev ?? 1.5;

    // Resolve percentiles (fallback to normal distribution if not provided)
    const p10 = percentile_10 ?? Math.max(0, er - 1.282 * sigma);
    const p25 = percentile_25 ?? Math.max(0, er - 0.674 * sigma);
    const p50 = er;
    const p75 = percentile_75 ?? er + 0.674 * sigma;
    const p90 = percentile_90 ?? er + 1.282 * sigma;

    // Domain with padding
    const lo  = Math.min(p10, actual_runs ?? Infinity, 0);
    const hi  = Math.max(p90, actual_runs ?? -Infinity);
    const pad = (hi - lo) * 0.12;
    const xMin = Math.max(0, lo - pad);
    const xMax = hi + pad;

    const x = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // ── Audit 2.6: KDE Density Curve (asymmetric, right-skewed shape) ─────────
    // Density derived from interval probability masses — shows actual skew.
    // Intervals: [<P10]=10%, [P10-P25]=15%, [P25-P50]=25%, [P50-P75]=25%, [P75-P90]=15%, [>P90]=10%
    const kdeAmplitude = midY * 0.78; // max violin height in pixels

    // Build density control points: (run_value, density)
    // density ≈ probability_mass / interval_width
    const intervals = [
      { lo: Math.max(0, p10 - (p25 - p10)), hi: p10, mass: 0.10 },
      { lo: p10, hi: p25,  mass: 0.15 },
      { lo: p25, hi: p50,  mass: 0.25 },
      { lo: p50, hi: p75,  mass: 0.25 },
      { lo: p75, hi: p90,  mass: 0.15 },
      { lo: p90, hi: p90 + (p90 - p75), mass: 0.10 },
    ];

    const densityPts: Array<[number, number]> = [];
    for (const seg of intervals) {
      const gap = seg.hi - seg.lo;
      const dens = gap > 0 ? seg.mass / gap : 0;
      densityPts.push([(seg.lo + seg.hi) / 2, dens]);
    }
    // anchor zeros at the edges
    densityPts.unshift([xMin, 0]);
    densityPts.push([xMax, 0]);

    const maxDens = Math.max(...densityPts.map(d => d[1]));

    // Scale density to pixel height
    const yScale = (dens: number) => (dens / maxDens) * kdeAmplitude;

    // Build the KDE curve path points (upper half only, since we want a density curve not full violin)
    const kdeLine = d3.line<[number, number]>()
      .x(d => x(d[0]))
      .y(d => midY - yScale(d[1]))
      .curve(d3.curveCatmullRom.alpha(0.5));

    // KDE filled area
    const kdeArea = d3.area<[number, number]>()
      .x(d => x(d[0]))
      .y0(midY)
      .y1(d => midY - yScale(d[1]))
      .curve(d3.curveCatmullRom.alpha(0.5));

    g.append('path')
      .datum(densityPts)
      .attr('d', kdeArea)
      .attr('fill', `${uColor}18`)
      .attr('stroke', 'none');

    g.append('path')
      .datum(densityPts)
      .attr('d', kdeLine)
      .attr('fill', 'none')
      .attr('stroke', `${uColor}70`)
      .attr('stroke-width', 1.5);

    // "KDE" label
    g.append('text')
      .attr('x', 2).attr('y', midY - kdeAmplitude + 4)
      .attr('fill', 'rgba(255,255,255,0.2)').attr('font-size', '7px')
      .attr('font-family', 'sans-serif').text('KDE');

    // ── Baseline (center line) ────────────────────────────────────────────────
    g.append('line')
      .attr('x1', 0).attr('x2', innerW)
      .attr('y1', midY).attr('y2', midY)
      .attr('stroke', 'rgba(255,255,255,0.08)')
      .attr('stroke-width', 1);

    // ── Whiskers (P10 – P90) ─────────────────────────────────────────────────
    const p10x = x(p10);
    const p90x = x(p90);

    g.append('line')
      .attr('x1', p10x).attr('x2', p90x)
      .attr('y1', midY).attr('y2', midY)
      .attr('stroke', 'rgba(255,255,255,0.35)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3');

    // P10 tick + label
    g.append('line')
      .attr('x1', p10x).attr('x2', p10x)
      .attr('y1', midY - 6).attr('y2', midY + 6)
      .attr('stroke', 'rgba(255,255,255,0.4)').attr('stroke-width', 1.5);
    g.append('text')
      .attr('x', p10x).attr('y', midY + 16)
      .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.35)')
      .attr('font-size', '8px').attr('font-family', 'JetBrains Mono, monospace')
      .text(`P10:${p10.toFixed(1)}`);

    // P90 tick + label
    g.append('line')
      .attr('x1', p90x).attr('x2', p90x)
      .attr('y1', midY - 6).attr('y2', midY + 6)
      .attr('stroke', 'rgba(255,255,255,0.4)').attr('stroke-width', 1.5);
    g.append('text')
      .attr('x', p90x).attr('y', midY + 16)
      .attr('text-anchor', 'middle').attr('fill', 'rgba(255,255,255,0.35)')
      .attr('font-size', '8px').attr('font-family', 'JetBrains Mono, monospace')
      .text(`P90:${p90.toFixed(1)}`);

    // ── IQR Box (P25 – P75) ───────────────────────────────────────────────────
    const p25x = x(p25);
    const p75x = x(p75);
    const boxY  = midY - boxH / 2;

    g.append('rect')
      .attr('x', p25x).attr('y', boxY)
      .attr('width', Math.max(0, p75x - p25x))
      .attr('height', boxH)
      .attr('fill', `${uColor}28`)
      .attr('stroke', uColor)
      .attr('stroke-width', 1.5)
      .attr('rx', 2);

    // P25/P75 labels inside box
    g.append('text')
      .attr('x', p25x + 3).attr('y', midY + boxH / 2 + 9)
      .attr('fill', 'rgba(255,255,255,0.35)').attr('font-size', '7px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .text(`${p25.toFixed(1)}`);
    g.append('text')
      .attr('x', p75x - 3).attr('y', midY + boxH / 2 + 9)
      .attr('text-anchor', 'end').attr('fill', 'rgba(255,255,255,0.35)')
      .attr('font-size', '7px').attr('font-family', 'JetBrains Mono, monospace')
      .text(`${p75.toFixed(1)}`);

    // ── E[R] mean marker ───────────────────────────────────────────────────────
    const erX = x(p50);
    g.append('circle')
      .attr('cx', erX).attr('cy', midY)
      .attr('r', 5)
      .attr('fill', '#00FF87')
      .attr('stroke', '#0A0E1A').attr('stroke-width', 2);

    g.append('text')
      .attr('x', erX).attr('y', boxY - 6)
      .attr('text-anchor', 'middle').attr('fill', '#00FF87')
      .attr('font-size', '11px').attr('font-weight', '700')
      .attr('font-family', 'JetBrains Mono, monospace')
      .text(`${p50.toFixed(2)}`);

    // ── Actual runs marker (post-game only) ────────────────────────────────────
    if (actual_runs !== undefined) {
      const arX = x(actual_runs);
      const arColor = '#F97316';
      g.append('line')
        .attr('x1', arX).attr('x2', arX)
        .attr('y1', boxY - 4).attr('y2', midY + boxH / 2 + 4)
        .attr('stroke', arColor).attr('stroke-width', 2)
        .attr('stroke-dasharray', '3,2');
      g.append('polygon')
        .attr('points', `${arX},${boxY - 10} ${arX - 5},${boxY - 2} ${arX + 5},${boxY - 2}`)
        .attr('fill', arColor);
      g.append('text')
        .attr('x', arX).attr('y', midY + boxH / 2 + 18)
        .attr('text-anchor', 'middle').attr('fill', arColor)
        .attr('font-size', '9px').attr('font-family', 'JetBrains Mono, monospace')
        .text(`Real:${actual_runs}`);
    }

    // ── X axis ────────────────────────────────────────────────────────────────
    const ticks = d3.range(Math.ceil(xMin), Math.floor(xMax) + 1);
    const axisG = g.append('g').attr('transform', `translate(0,${innerH})`);
    axisG.append('line').attr('x1', 0).attr('x2', innerW)
      .attr('stroke', 'rgba(255,255,255,0.1)');
    ticks.forEach((t) => {
      if (t < xMin || t > xMax) return;
      axisG.append('text')
        .attr('x', x(t)).attr('y', 12).attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.3)').attr('font-size', '9px')
        .attr('font-family', 'JetBrains Mono, monospace').text(t);
    });

    // League average dashed ref
    const la = 4.5;
    if (la >= xMin && la <= xMax) {
      g.append('line')
        .attr('x1', x(la)).attr('x2', x(la))
        .attr('y1', 0).attr('y2', innerH)
        .attr('stroke', 'rgba(255,255,255,0.15)')
        .attr('stroke-width', 1).attr('stroke-dasharray', '4,3');
      g.append('text')
        .attr('x', x(la) + 3).attr('y', 10)
        .attr('fill', 'rgba(255,255,255,0.25)').attr('font-size', '7px')
        .attr('font-family', 'JetBrains Mono, monospace').text('Liga 4.5');
    }
  }, [er, std_dev, percentile_10, percentile_25, percentile_75, percentile_90,
      uncertainty_level, actual_runs]);

  return (
    <div className={className}>
      <div ref={containerRef} style={{ width: '100%' }}>
        <svg ref={svgRef} style={{ width: '100%', height: '155px' }} />
      </div>

      {/* Uncertainty badge — colour + symbol (WCAG AA) */}
      <div className="mt-2 flex items-center gap-2 flex-wrap">
        <span
          className="inline-flex items-center gap-1 text-[10px] font-mono font-bold px-2 py-0.5 rounded border"
          style={{
            color: uColor,
            borderColor: `${uColor}55`,
            background: `${uColor}11`,
          }}
        >
          <span>{UNCERTAINTY_SYMBOL[uncertainty_level]}</span>
          {uncertainty_level.toUpperCase()}
        </span>
        <span className="text-[10px] text-slate-500">
          {UNCERTAINTY_LABEL[uncertainty_level]}
        </span>
        {std_dev !== undefined && (
          <span className="text-[10px] font-mono text-slate-400">
            σ = {std_dev.toFixed(2)} car.
          </span>
        )}
        {simulation_n !== undefined && simulation_n > 0 && (
          <span className="text-[10px] text-slate-600">
            n = {simulation_n.toLocaleString()} sims
          </span>
        )}
      </div>
    </div>
  );
}
