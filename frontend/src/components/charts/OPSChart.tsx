/**
 * OPSChart — Roadmap 1.5
 *
 * Replaced the 4-color rainbow palette with a 2-tone diverging scale
 * anchored to the MLB league average OPS (~0.750).
 *
 * Color scheme: blue-orange (colorblind-safe for deuteranopia/protanopia).
 *   - Above league avg → blue shades (intensity = distance from avg)
 *   - Below league avg → amber/orange shades (intensity = distance from avg)
 *   - At league avg (±0.020) → neutral gray
 *
 * Added: a text position marker (+/-) relative to league avg so the chart
 * is readable in grayscale without relying on color alone.
 */
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Cell,
  ReferenceLine,
  LabelList,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import type { Player } from '../../types';

interface OPSChartProps {
  lineup: Player[];
}

// MLB league avg OPS 2024 (~.750). Bands:
// Strong above: ≥ +0.100 → deep blue
// Above: ≥ +0.040 → mid blue
// Slightly above: ≥ +0.010 → light blue
// At avg: < ±0.030 → gray
// Slightly below: < -0.010 → light amber
// Below: < -0.040 → mid amber/orange
// Strong below: < -0.100 → deep orange
const LEAGUE_AVG_OPS = 0.750;

function getOPSColor(ops: number): string {
  const delta = ops - LEAGUE_AVG_OPS;
  if (delta >= 0.100) return '#1D4ED8'; // deep blue
  if (delta >= 0.040) return '#3B82F6'; // mid blue
  if (delta >= 0.010) return '#93C5FD'; // light blue
  if (delta >= -0.010) return '#6B7280'; // neutral gray
  if (delta >= -0.040) return '#FCD34D'; // light amber
  if (delta >= -0.100) return '#F97316'; // orange
  return '#DC2626'; // deep red-orange (well below avg)
}

/** Accessibility: text label showing position relative to league avg */
function getPositionLabel(ops: number): string {
  const delta = ops - LEAGUE_AVG_OPS;
  if (delta >= 0.100) return '↑↑↑';
  if (delta >= 0.040) return '↑↑';
  if (delta >= 0.010) return '↑';
  if (delta >= -0.010) return '≈ Liga';
  if (delta >= -0.040) return '↓';
  if (delta >= -0.100) return '↓↓';
  return '↓↓↓';
}

function getLastName(name: string): string {
  const parts = name.trim().split(/\s+/);
  return parts[parts.length - 1];
}

interface CustomLabelProps {
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  value?: number;
}

function CustomLabel({ x = 0, y = 0, width = 0, height = 0, value = 0 }: CustomLabelProps) {
  const color = getOPSColor(value);
  const posLabel = getPositionLabel(value);
  return (
    <text
      x={x + width + 6}
      y={y + height / 2 + 1}
      fill={color}
      fontSize={10}
      fontFamily="JetBrains Mono, monospace"
      dominantBaseline="middle"
    >
      {value.toFixed(3)} {posLabel}
    </text>
  );
}

export default function OPSChart({ lineup }: OPSChartProps) {
  const data = lineup.map((p) => ({
    name: `${p.order}. ${getLastName(p.name)}`,
    ops: p.ops,
    color: getOPSColor(p.ops),
  }));

  return (
    <div>
      <p className="text-xs uppercase tracking-widest text-slate-500 mb-1">
        OPS por Posición — Vs. liga ({LEAGUE_AVG_OPS.toFixed(3)})
      </p>
      {/* Colorblind-safe legend — uses symbols, not color alone */}
      <p style={{ fontSize: '0.6rem', color: '#374151', marginBottom: 8, fontFamily: 'JetBrains Mono, monospace' }}>
        ↑↑↑ elite &nbsp;·&nbsp; ↑↑ sobre avg &nbsp;·&nbsp; ↑ ligeramente &nbsp;·&nbsp;
        ≈ Liga avg &nbsp;·&nbsp; ↓ bajo &nbsp;·&nbsp; ↓↓ debajo &nbsp;·&nbsp; ↓↓↓ muy bajo
      </p>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 110, left: 4, bottom: 4 }}
        >
          <XAxis
            type="number"
            domain={[0.4, 1.1]}
            tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
            axisLine={{ stroke: 'rgba(255,255,255,0.1)' }}
            tickLine={false}
            tickCount={8}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={72}
            tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11, fontFamily: 'Inter, sans-serif' }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: 'rgba(255,255,255,0.04)' }}
            contentStyle={{
              background: '#111827',
              border: '1px solid #1F2937',
              borderRadius: 6,
              color: '#E8EAF6',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: 12,
            }}
            formatter={(value: number) => [
              `${value.toFixed(3)}  (Δ ${(value - LEAGUE_AVG_OPS >= 0 ? '+' : '')}${(value - LEAGUE_AVG_OPS).toFixed(3)} vs liga)`,
              'OPS',
            ]}
          />
          {/* League average reference line */}
          <ReferenceLine
            x={LEAGUE_AVG_OPS}
            stroke="rgba(255,255,255,0.4)"
            strokeDasharray="4 3"
            label={{
              value: `Liga ${LEAGUE_AVG_OPS.toFixed(3)}`,
              position: 'top',
              fill: 'rgba(255,255,255,0.40)',
              fontSize: 9,
              fontFamily: 'JetBrains Mono, monospace',
            }}
          />
          <Bar dataKey="ops" radius={[0, 3, 3, 0]} maxBarSize={18}>
            {data.map((entry, idx) => (
              <Cell key={idx} fill={entry.color} opacity={0.88} />
            ))}
            <LabelList dataKey="ops" content={<CustomLabel />} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
