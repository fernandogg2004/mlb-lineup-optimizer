import { useRef, useEffect } from 'react';
import * as d3 from 'd3';

interface RunsDistributionProps {
  er: number;
  percentiles: Record<string, number>;
  /** Explicit P10/P50/P90 for marker lines (Roadmap 1.6 + 2.3) */
  p10?: number;
  p50?: number;
  p90?: number;
}

const PERCENTILE_MAP: [string, number][] = [
  ['1', 0.01],
  ['2', 0.02],
  ['5', 0.05],
  ['25', 0.25],
  ['50', 0.50],
  ['75', 0.75],
  ['95', 0.95],
  ['98', 0.98],
  ['99', 0.99],
];

export default function RunsDistribution({ er, percentiles, p10: p10Prop, p50: p50Prop, p90: p90Prop }: RunsDistributionProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const p25 = percentiles['25'];
  const p75 = percentiles['75'];
  // Use explicit props if provided, fall back to percentiles dict
  const p10Marker = p10Prop ?? percentiles['10'];
  const p50Marker = p50Prop ?? percentiles['50'] ?? er;
  const p90Marker = p90Prop ?? percentiles['90'];

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = containerRef.current.clientWidth || 400;
    const height = 160;
    const margin = { top: 20, right: 16, bottom: 24, left: 8 };
    const innerW = width - margin.left - margin.right;
    const innerH = height - margin.top - margin.bottom;

    // Build CDF points from percentiles
    const cdfPoints: [number, number][] = [];
    for (const [key, prob] of PERCENTILE_MAP) {
      const val = percentiles[key];
      if (val !== undefined) {
        cdfPoints.push([val, prob]);
      }
    }

    // Fallback: normal approx if not enough points
    let xMin = 0;
    let xMax = 12;

    if (cdfPoints.length >= 3) {
      xMin = Math.max(0, cdfPoints[0][0] - 0.5);
      xMax = Math.min(16, cdfPoints[cdfPoints.length - 1][0] + 1);
    } else {
      // Normal approx with sigma=1.5
      const mu = er;
      const sigma = 1.5;
      xMin = Math.max(0, mu - 3.5 * sigma);
      xMax = mu + 3.5 * sigma;
      for (let p = 0.05; p <= 0.95; p += 0.1) {
        cdfPoints.push([mu + sigma * normInv(p), p]);
      }
      cdfPoints.sort((a, b) => a[0] - b[0]);
    }

    // Derive PDF via finite differences
    const N = 80;
    const xs = d3.range(N).map((i) => xMin + (i / (N - 1)) * (xMax - xMin));

    // Interpolate CDF at each x
    const cdfInterp = d3
      .scaleLinear()
      .domain(cdfPoints.map((d) => d[0]))
      .range(cdfPoints.map((d) => d[1]))
      .clamp(true);

    const pdfRaw = xs.map((x, i) => {
      if (i === 0) return 0;
      const dx = xs[i] - xs[i - 1];
      const dP = cdfInterp(xs[i]) - cdfInterp(xs[i - 1]);
      return Math.max(0, dP / dx);
    });

    // Normalize area
    const totalArea = pdfRaw.reduce((sum, v, i) => {
      if (i === 0) return sum;
      return sum + v * (xs[i] - xs[i - 1]);
    }, 0);
    const pdf = totalArea > 0 ? pdfRaw.map((v) => v / totalArea) : pdfRaw;

    // Scales
    const xScale = d3.scaleLinear().domain([xMin, xMax]).range([0, innerW]);
    const yMax = d3.max(pdf) ?? 1;
    const yScale = d3.scaleLinear().domain([0, yMax * 1.1]).range([innerH, 0]);

    // Gradient definition
    const defs = svg.append('defs');
    const gradientId = 'runs-gradient';
    const grad = defs
      .append('linearGradient')
      .attr('id', gradientId)
      .attr('x1', '0%')
      .attr('x2', '0%')
      .attr('y1', '0%')
      .attr('y2', '100%');
    grad.append('stop').attr('offset', '0%').attr('stop-color', '#3B82F6').attr('stop-opacity', 0.4);
    grad.append('stop').attr('offset', '100%').attr('stop-color', '#3B82F6').attr('stop-opacity', 0);

    const g = svg
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Shaded IQR band (p25–p75)
    if (p25 !== undefined && p75 !== undefined) {
      g.append('rect')
        .attr('x', xScale(p25))
        .attr('width', Math.max(0, xScale(p75) - xScale(p25)))
        .attr('y', 0)
        .attr('height', innerH)
        .attr('fill', 'rgba(59,130,246,0.12)');
    }

    // PDF area
    const areaGen = d3
      .area<number>()
      .x((_, i) => xScale(xs[i]))
      .y0(innerH)
      .y1((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));

    g.append('path')
      .datum(pdf)
      .attr('fill', `url(#${gradientId})`)
      .attr('d', areaGen);

    // PDF curve
    const lineGen = d3
      .line<number>()
      .x((_, i) => xScale(xs[i]))
      .y((d) => yScale(d))
      .curve(d3.curveCatmullRom.alpha(0.5));

    g.append('path')
      .datum(pdf)
      .attr('fill', 'none')
      .attr('stroke', '#3B82F6')
      .attr('stroke-width', 2)
      .attr('d', lineGen);

    // League average dashed line at 4.50
    const leagueAvg = 4.5;
    if (leagueAvg >= xMin && leagueAvg <= xMax) {
      g.append('line')
        .attr('x1', xScale(leagueAvg))
        .attr('x2', xScale(leagueAvg))
        .attr('y1', 0)
        .attr('y2', innerH)
        .attr('stroke', 'rgba(255,255,255,0.35)')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,3');

      g.append('text')
        .attr('x', xScale(leagueAvg) + 3)
        .attr('y', 10)
        .attr('fill', 'rgba(255,255,255,0.4)')
        .attr('font-size', '9px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text('Liga 4.50');
    }

    // P10 / P90 marker lines (Roadmap 2.3 — floor/ceiling visible)
    for (const { val, label, color } of [
      { val: p10Marker, label: 'P10', color: 'rgba(156,163,175,0.6)' },
      { val: p90Marker, label: 'P90', color: 'rgba(156,163,175,0.6)' },
    ]) {
      if (val === undefined || val < xMin || val > xMax) continue;
      const px = xScale(val);
      g.append('line')
        .attr('x1', px).attr('x2', px)
        .attr('y1', 0).attr('y2', innerH)
        .attr('stroke', color)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');
      g.append('text')
        .attr('x', px + 2).attr('y', innerH - 4)
        .attr('fill', color).attr('font-size', '8px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(label);
    }

    // P50 (median) marker
    if (p50Marker !== undefined && p50Marker >= xMin && p50Marker <= xMax) {
      const p50x = xScale(p50Marker);
      g.append('line')
        .attr('x1', p50x).attr('x2', p50x)
        .attr('y1', 0).attr('y2', innerH)
        .attr('stroke', 'rgba(96,165,250,0.7)')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,2');
      g.append('text')
        .attr('x', p50x + 2).attr('y', 20)
        .attr('fill', 'rgba(96,165,250,0.8)').attr('font-size', '8px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(`P50 ${p50Marker.toFixed(1)}`);
    }

    // E[R] vertical line
    const erX = xScale(er);
    const erLabel = `E[R] ${er.toFixed(2)}`;
    const erLabelW = erLabel.length * 6.5; // approx px width at 10px mono
    const erLabelX = Math.min(Math.max(erX + 3, 2), innerW - erLabelW - 2);

    g.append('line')
      .attr('x1', erX)
      .attr('x2', erX)
      .attr('y1', 0)
      .attr('y2', innerH)
      .attr('stroke', '#00FF87')
      .attr('stroke-width', 2);

    // Opaque background rect so label is readable over the curve
    g.append('rect')
      .attr('x', erLabelX - 2)
      .attr('y', 0)
      .attr('width', erLabelW + 4)
      .attr('height', 14)
      .attr('fill', '#0D1117')
      .attr('opacity', 0.85)
      .attr('rx', 2);

    g.append('text')
      .attr('x', erLabelX)
      .attr('y', 10)
      .attr('fill', '#00FF87')
      .attr('font-size', '10px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .attr('font-weight', '700')
      .text(erLabel);

    // X axis
    const xTicks = d3.range(Math.ceil(xMin), Math.floor(xMax) + 1, 1);
    const xAxis = g.append('g').attr('transform', `translate(0,${innerH})`);

    xAxis
      .append('line')
      .attr('x1', 0)
      .attr('x2', innerW)
      .attr('stroke', 'rgba(255,255,255,0.15)');

    xTicks.forEach((tick) => {
      if (tick < xMin || tick > xMax) return;
      xAxis
        .append('text')
        .attr('x', xScale(tick))
        .attr('y', 14)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.35)')
        .attr('font-size', '9px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(tick);
    });
  }, [er, percentiles, p10Marker, p50Marker, p90Marker]);

  return (
    <div>
      <p className="text-xs uppercase tracking-widest text-slate-500 mb-2">
        Distribución de Carreras Esperadas
      </p>
      <div ref={containerRef} style={{ width: '100%' }}>
        <svg ref={svgRef} style={{ width: '100%', height: '160px' }} />
      </div>
      {p25 !== undefined && p75 !== undefined && (
        <p className="text-xs text-slate-400 mt-1 font-mono">
          Concentración 50%: {p25.toFixed(1)} – {p75.toFixed(1)} car.
        </p>
      )}
    </div>
  );
}

// Error function approximation
function erf(x: number): number {
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

// Inverse normal CDF approximation (Beasley-Springer-Moro)
function normInv(p: number): number {
  const a = [
    -3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2,
    1.38357751867269e2, -3.066479806614716e1, 2.506628277459239,
  ];
  const b = [
    -5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2,
    6.680131188771972e1, -1.328068155288572e1,
  ];
  const c = [
    -7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838,
    -2.549732539343734, 4.374664141464968, 2.938163982698783,
  ];
  const d = [
    7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996,
    3.754408661907416,
  ];
  const pLow = 0.02425;
  const pHigh = 1 - pLow;

  if (p < pLow) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (
      (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  } else if (p <= pHigh) {
    const q = p - 0.5;
    const r = q * q;
    return (
      ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
      (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    );
  } else {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return (
      -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    );
  }
}
