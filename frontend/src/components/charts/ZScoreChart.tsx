/**
 * ZScoreChart — Roadmap 1.1
 *
 * Replaces the radar chart with horizontal Z-score deviation bars.
 * Each metric shows distance from the league mean in standard deviations.
 *
 * Layout: one row per metric (OBP, SLG, ISO, wOBA, K%)
 *   - Center line = league average (0)
 *   - Bands at ±1σ and ±2σ
 *   - K% axis is INVERTED (lower K% is better → positive = good)
 *   - Value labeled at the right edge
 *
 * Design principle: "¿está este ISO por encima de la media y cuánto?"
 * readable in under 1 second, no legend needed.
 */
import type { Player } from '../../types';

interface ZScoreChartProps {
  lineup: Player[];
  pitcherHand?: string;
  pitcherName?: string;
}

interface Metric {
  key: string;
  label: string;
  leagueMu: number;
  leagueSigma: number;
  computeFn: (p: Player) => number;
  /** If true, lower raw value = better → invert Z-score sign for display */
  invertedAxis?: boolean;
  formatValue: (v: number) => string;
}

const METRICS: Metric[] = [
  {
    key: 'obp',
    label: 'OBP',
    leagueMu: 0.318,
    leagueSigma: 0.020,
    computeFn: (p) => p.obp,
    formatValue: (v) => v.toFixed(3),
  },
  {
    key: 'slg',
    label: 'SLG',
    leagueMu: 0.413,
    leagueSigma: 0.045,
    computeFn: (p) => p.avg + p.iso,
    formatValue: (v) => v.toFixed(3),
  },
  {
    key: 'iso',
    label: 'ISO',
    leagueMu: 0.165,
    leagueSigma: 0.030,
    computeFn: (p) => p.iso,
    formatValue: (v) => v.toFixed(3),
  },
  {
    key: 'woba',
    label: 'wOBA',
    leagueMu: 0.315,
    leagueSigma: 0.025,
    computeFn: (p) => p.woba,
    formatValue: (v) => v.toFixed(3),
  },
  {
    key: 'kpct',
    label: 'K%',
    leagueMu: 0.230,
    leagueSigma: 0.040,
    // Inverted: low K% is good → positive bar = good
    computeFn: (p) => Math.max(0, Math.min(1, 1 - p.avg * 3.5)),
    invertedAxis: true,
    formatValue: (v) => `${(v * 100).toFixed(1)}%`,
  },
];

function avg(values: number[]): number {
  if (!values.length) return 0;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function zScore(value: number, mu: number, sigma: number, inverted = false): number {
  const z = (value - mu) / sigma;
  return inverted ? -z : z;
}

function barColor(z: number): string {
  const abs = Math.abs(z);
  if (z > 0) {
    return abs > 1.5 ? '#00FF87' : abs > 0.5 ? '#34D399' : '#6EE7B7';
  } else {
    return abs > 1.5 ? '#EF4444' : abs > 0.5 ? '#F87171' : '#FCA5A5';
  }
}

export default function ZScoreChart({ lineup, pitcherHand, pitcherName }: ZScoreChartProps) {
  if (!lineup.length) return null;

  // Max Z to display on each side (clip at 3σ)
  const MAX_Z = 3;
  const BAR_WIDTH = 180; // px total (half each side)
  const HALF = BAR_WIDTH / 2;

  return (
    <div>
      {/* Header */}
      <div style={{ marginBottom: 12 }}>
        <p style={{
          fontSize: '0.63rem', fontWeight: 700, letterSpacing: '0.14em',
          textTransform: 'uppercase', color: '#2E5580', marginBottom: 2,
        }}>
          Perfil Ofensivo vs Liga (Z-score)
        </p>
        {pitcherName && (
          <p style={{ fontSize: '0.7rem', color: '#546E7A' }}>
            vs {pitcherName} ({pitcherHand}HP)
          </p>
        )}
      </div>

      {/* Band labels */}
      <div style={{
        display: 'flex', alignItems: 'center', marginBottom: 4,
        paddingLeft: 48,
      }}>
        <div style={{ width: BAR_WIDTH, position: 'relative', height: 14 }}>
          {[-2, -1, 0, 1, 2].map(z => (
            <span key={z} style={{
              position: 'absolute',
              left: HALF + z * (HALF / MAX_Z) - 8,
              fontSize: '0.58rem', color: '#374151',
              fontFamily: 'JetBrains Mono, monospace',
            }}>
              {z === 0 ? 'μ' : `${z > 0 ? '+' : ''}${z}σ`}
            </span>
          ))}
        </div>
      </div>

      {/* Metric rows */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {METRICS.map((m) => {
          const rawTeam = avg(lineup.map(m.computeFn));
          const z = zScore(rawTeam, m.leagueMu, m.leagueSigma, m.invertedAxis ?? false);
          const zClamped = Math.max(-MAX_Z, Math.min(MAX_Z, z));
          const barPx = Math.abs(zClamped) * (HALF / MAX_Z);
          const color = barColor(z);
          const isPositive = z >= 0;

          return (
            <div key={m.key} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              {/* Metric label */}
              <div style={{
                width: 44, textAlign: 'right',
                fontSize: '0.72rem', fontFamily: 'JetBrains Mono, monospace',
                color: '#6B7280', flexShrink: 0,
              }}>
                {m.label}
                {m.invertedAxis && (
                  <span style={{ color: '#374151', fontSize: '0.58rem' }}>↓</span>
                )}
              </div>

              {/* Bar track */}
              <div style={{
                width: BAR_WIDTH, height: 18, position: 'relative',
                background: '#0D1117', borderRadius: 4, flexShrink: 0,
              }}>
                {/* ±1σ band background */}
                <div style={{
                  position: 'absolute',
                  left: HALF - HALF / MAX_Z,
                  width: 2 * HALF / MAX_Z,
                  height: '100%',
                  background: 'rgba(255,255,255,0.04)',
                  borderRadius: 2,
                }} />

                {/* Center line (league avg) */}
                <div style={{
                  position: 'absolute', left: HALF - 0.5, width: 1,
                  height: '100%', background: 'rgba(255,255,255,0.25)',
                }} />

                {/* ±1σ lines */}
                {[-1, 1].map(s => (
                  <div key={s} style={{
                    position: 'absolute',
                    left: HALF + s * (HALF / MAX_Z) - 0.5,
                    width: 1, height: '100%',
                    background: 'rgba(255,255,255,0.12)',
                  }} />
                ))}

                {/* ±2σ lines */}
                {[-2, 2].map(s => (
                  <div key={s} style={{
                    position: 'absolute',
                    left: HALF + s * (HALF / MAX_Z) - 0.5,
                    width: 1, height: '100%',
                    background: 'rgba(255,255,255,0.07)',
                  }} />
                ))}

                {/* Actual bar */}
                <div style={{
                  position: 'absolute',
                  left: isPositive ? HALF : HALF - barPx,
                  width: barPx,
                  height: '100%',
                  background: color,
                  borderRadius: isPositive ? '0 3px 3px 0' : '3px 0 0 3px',
                  opacity: 0.85,
                  transition: 'width 0.4s ease',
                }} />
              </div>

              {/* Value label */}
              <span style={{
                fontSize: '0.72rem', fontFamily: 'JetBrains Mono, monospace',
                color, minWidth: 44,
              }}>
                {m.formatValue(rawTeam)}
              </span>

              {/* Z-score badge */}
              <span style={{
                fontSize: '0.60rem', fontFamily: 'JetBrains Mono, monospace',
                color: '#374151', minWidth: 36,
              }}>
                {z >= 0 ? '+' : ''}{z.toFixed(2)}σ
              </span>
            </div>
          );
        })}
      </div>

      {/* League note */}
      <p style={{
        marginTop: 10, fontSize: '0.62rem', color: '#1F2937',
        fontFamily: 'JetBrains Mono, monospace',
      }}>
        μ = media de liga · ±1σ / ±2σ bandas de referencia ·
        K% eje invertido (↓ = mejor)
      </p>
    </div>
  );
}
