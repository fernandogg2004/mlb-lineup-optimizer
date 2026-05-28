import { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import type { Player } from '../../types';

interface RadarChartProps {
  lineup: Player[];
  pitcherHand: string;
  pitcherName: string;
}

interface RadarVar {
  key: string;
  label: string;
  leagueVal: number;
  computeFn: (p: Player) => number;
  range: [number, number];
}

const VARS: RadarVar[] = [
  {
    key: 'OBP',
    label: 'OBP',
    leagueVal: 0.318,
    computeFn: (p) => p.obp,
    range: [0.25, 0.45],
  },
  {
    key: 'SLG',
    label: 'SLG',
    leagueVal: 0.413,
    computeFn: (p) => p.avg + p.iso,
    range: [0.3, 0.65],
  },
  {
    key: 'ISO',
    label: 'ISO',
    leagueVal: 0.165,
    computeFn: (p) => p.iso,
    range: [0.05, 0.35],
  },
  {
    key: 'K%',
    label: 'K%',
    leagueVal: 0.23,
    computeFn: (p) => Math.max(0, Math.min(1, 1 - p.avg * 3.5)),
    range: [0.05, 0.45],
  },
  {
    key: 'BB%',
    label: 'BB%',
    leagueVal: 0.08,
    computeFn: (p) => Math.max(0, p.obp - p.avg),
    range: [0.02, 0.18],
  },
];

function avg(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function polarToCartesian(
  cx: number,
  cy: number,
  r: number,
  angle: number
): [number, number] {
  return [cx + r * Math.cos(angle), cy + r * Math.sin(angle)];
}

export default function RadarChart({ lineup, pitcherHand, pitcherName }: RadarChartProps) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || lineup.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const size = 280;
    const cx = size / 2;
    const cy = size / 2;
    const radius = 100;
    const n = VARS.length;
    const angleStep = (2 * Math.PI) / n;

    // Compute team values
    const teamValues = VARS.map((v) => avg(lineup.map(v.computeFn)));
    const leagueValues = VARS.map((v) => v.leagueVal);

    // Normalize to 0-1 within range
    function norm(val: number, range: [number, number]): number {
      return Math.max(0, Math.min(1, (val - range[0]) / (range[1] - range[0])));
    }

    const teamNorm = VARS.map((v, i) => norm(teamValues[i], v.range));
    const leagueNorm = VARS.map((v, i) => norm(leagueValues[i], v.range));

    svg.attr('width', size).attr('height', size);

    const g = svg.append('g');

    // Concentric rings
    const rings = [0.25, 0.5, 0.75, 1.0];
    for (const frac of rings) {
      const pts = d3.range(n).map((i) => {
        const angle = -Math.PI / 2 + i * angleStep;
        return polarToCartesian(cx, cy, radius * frac, angle);
      });
      g.append('polygon')
        .attr('points', pts.map((p) => p.join(',')).join(' '))
        .attr('fill', 'none')
        .attr('stroke', 'rgba(255,255,255,0.08)')
        .attr('stroke-width', 1);

      // Ring value label at 12-o'clock
      const labelVal = VARS[0].range[0] + frac * (VARS[0].range[1] - VARS[0].range[0]);
      g.append('text')
        .attr('x', cx)
        .attr('y', cy - radius * frac - 3)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.2)')
        .attr('font-size', '8px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(labelVal.toFixed(2));
    }

    // Axis lines
    VARS.forEach((_, i) => {
      const angle = -Math.PI / 2 + i * angleStep;
      const [x, y] = polarToCartesian(cx, cy, radius, angle);
      g.append('line')
        .attr('x1', cx)
        .attr('y1', cy)
        .attr('x2', x)
        .attr('y2', y)
        .attr('stroke', 'rgba(255,255,255,0.1)')
        .attr('stroke-width', 1);
    });

    // League polygon
    const leaguePts = d3.range(n).map((i) => {
      const angle = -Math.PI / 2 + i * angleStep;
      return polarToCartesian(cx, cy, radius * leagueNorm[i], angle);
    });
    g.append('polygon')
      .attr('points', leaguePts.map((p) => p.join(',')).join(' '))
      .attr('fill', 'none')
      .attr('stroke', 'rgba(255,255,255,0.6)')
      .attr('stroke-width', 1.5)
      .attr('stroke-dasharray', '4,3');

    // League vertex dots
    leaguePts.forEach(([x, y]) => {
      g.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 3)
        .attr('fill', 'rgba(255,255,255,0.6)');
    });

    // Team polygon
    const teamPts = d3.range(n).map((i) => {
      const angle = -Math.PI / 2 + i * angleStep;
      return polarToCartesian(cx, cy, radius * teamNorm[i], angle);
    });
    g.append('polygon')
      .attr('points', teamPts.map((p) => p.join(',')).join(' '))
      .attr('fill', '#00FF87')
      .attr('fill-opacity', 0.08)
      .attr('stroke', '#00FF87')
      .attr('stroke-width', 2.5);

    // Team vertex dots
    teamPts.forEach(([x, y]) => {
      g.append('circle')
        .attr('cx', x)
        .attr('cy', y)
        .attr('r', 4)
        .attr('fill', '#00FF87');
    });

    // Axis labels
    VARS.forEach((v, i) => {
      const angle = -Math.PI / 2 + i * angleStep;
      const labelR = radius + 22;
      const [lx, ly] = polarToCartesian(cx, cy, labelR, angle);

      let anchor = 'middle';
      if (lx < cx - 10) anchor = 'end';
      else if (lx > cx + 10) anchor = 'start';

      const displayVal = teamValues[i];
      const formatted =
        v.key === 'K%' || v.key === 'BB%'
          ? `${(displayVal * 100).toFixed(1)}%`
          : displayVal.toFixed(3);

      g.append('text')
        .attr('x', lx)
        .attr('y', ly)
        .attr('text-anchor', anchor)
        .attr('dominant-baseline', 'middle')
        .attr('fill', 'rgba(255,255,255,0.9)')
        .attr('font-size', '11px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(`${v.label} ${formatted}`);
    });
  }, [lineup]);

  return (
    <div>
      <div className="mb-2">
        <p className="text-xs uppercase tracking-widest text-slate-500">
          Perfil Ofensivo: Lineup vs Liga
        </p>
        {pitcherName && (
          <p className="text-xs text-slate-400 mt-0.5">
            vs {pitcherName} ({pitcherHand}HP)
          </p>
        )}
      </div>
      <div className="flex justify-center">
        <svg ref={svgRef} width={280} height={280} />
      </div>
      {/* Legend */}
      <div className="flex items-center gap-4 mt-2 justify-center">
        <div className="flex items-center gap-1.5">
          <div className="w-5 h-0.5" style={{ background: '#00FF87' }} />
          <span className="text-xs text-slate-400">Lineup</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div
            className="w-5 h-0.5"
            style={{ background: 'rgba(255,255,255,0.6)', borderTop: '1px dashed' }}
          />
          <span className="text-xs text-slate-400">Liga</span>
        </div>
      </div>
    </div>
  );
}
