/**
 * MetricsPanel — Audit 2.4 redesign.
 *
 * Brier Score is the PRIMARY KPI (occupies ~60% of panel width).
 * Log-Loss is demoted to a secondary card.
 *
 * Layout:
 *   ┌───────────────────────────────────┬──────────────────┐
 *   │  BRIER SCORE  (hero, large font)  │  LOG-LOSS (sec.) │
 *   │  [30d] [90d] [Temporada]          │                  │
 *   │  0.226  ▼ mejor q/ benchmark      │  0.623           │
 *   │  ████████░░ progress bar          │  vs Elo baseline │
 *   └───────────────────────────────────┴──────────────────┘
 *
 * Temporal tabs: show brier_score_30d / brier_score_90d / brier_score_season
 * from RollingMetrics when available; otherwise show "—" with explanation.
 */
import { useState } from 'react';
import type { RollingMetrics } from '../../types';

interface MetricsPanelProps {
  metrics: RollingMetrics;
  /** Single-game log-loss — shown only in collapsed detail (Bug 5). */
  gameLogLoss?: number;
  className?: string;
}

// ── Brier benchmark (0.228 = MLB Elo baseline per audit §3.2) ─────────────────
const BRIER_BENCHMARK = 0.228;

type BrierPeriod = '30d' | '90d' | 'temporada';

// ── Sub-components ────────────────────────────────────────────────────────────

function PeriodTab({
  label,
  active,
  available,
  onClick,
}: {
  label: string;
  active: boolean;
  available: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      title={available ? undefined : 'Datos insuficientes (n < 30)'}
      style={{
        padding: '3px 10px',
        borderRadius: 6,
        fontSize: '0.65rem',
        fontWeight: 700,
        letterSpacing: '0.08em',
        border: active ? '1px solid #3B82F6' : '1px solid #1F2937',
        background: active ? 'rgba(59,130,246,0.15)' : 'transparent',
        color: active ? '#60A5FA' : available ? '#6B7280' : '#374151',
        cursor: available ? 'pointer' : 'not-allowed',
        transition: 'all 0.15s',
      }}
    >
      {label}
      {!available && (
        <span style={{ marginLeft: 3, opacity: 0.6 }}>—</span>
      )}
    </button>
  );
}

function TrendBar({
  value,
  benchmark,
}: {
  value: number;
  benchmark: number;
}) {
  // Scale: 0 = perfect Brier, max shown = benchmark × 1.4
  const maxScale = benchmark * 1.4;
  const modelPct = Math.min(98, (value / maxScale) * 100);
  const benchPct = Math.min(98, (benchmark / maxScale) * 100);
  const isBetter = value <= benchmark;

  return (
    <div style={{ position: 'relative', height: 8, borderRadius: 4, background: '#1F2937', overflow: 'visible' }}>
      {/* Model fill */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          height: 8,
          width: `${modelPct}%`,
          background: isBetter
            ? 'linear-gradient(90deg, #22C55E, #4ADE80)'
            : 'linear-gradient(90deg, #F59E0B, #FBBF24)',
          borderRadius: 4,
          transition: 'width 0.4s ease',
        }}
      />
      {/* Benchmark marker */}
      <div
        style={{
          position: 'absolute',
          top: -3,
          left: `${benchPct}%`,
          transform: 'translateX(-50%)',
          width: 2,
          height: 14,
          background: '#6B7280',
          borderRadius: 1,
        }}
        title={`Benchmark: ${benchmark.toFixed(3)}`}
      />
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function MetricsPanel({
  metrics,
  gameLogLoss,
  className = '',
}: MetricsPanelProps) {
  const [period, setPeriod] = useState<BrierPeriod>('temporada');
  const [showGameDetail, setShowGameDetail] = useState(false);

  // Resolve Brier value and n for selected period
  const brierByPeriod: Record<BrierPeriod, number | undefined> = {
    '30d': metrics.brier_score_30d,
    '90d': metrics.brier_score_90d,
    'temporada': metrics.brier_score_season,
  };
  const nByPeriod: Record<BrierPeriod, number | undefined> = {
    '30d': metrics.n_games_30d,
    '90d': metrics.n_games_90d,
    'temporada': metrics.n_games_evaluated,
  };

  const activeBrier = brierByPeriod[period] ?? metrics.brier_score_season;
  const activeN = nByPeriod[period] ?? metrics.n_games_evaluated;
  const benchmark = metrics.brier_score_benchmark ?? BRIER_BENCHMARK;
  const brierBetter = activeBrier <= benchmark;
  const brierDelta = activeBrier - benchmark;
  const brierDeltaStr = `${brierDelta >= 0 ? '+' : ''}${brierDelta.toFixed(3)}`;

  // Trend indicator
  const trend = metrics.brier_trend;
  const trendLabel = trend === undefined
    ? null
    : trend >= 0.005
    ? { text: '▲ Mejorando', color: '#22C55E' }
    : trend <= -0.005
    ? { text: '▼ Empeorando', color: '#EF4444' }
    : { text: '→ Estable', color: '#78909C' };

  const completionPct = metrics.n_games_season > 0
    ? (metrics.n_games_evaluated / metrics.n_games_season) * 100
    : 0;

  // Log-loss comparison
  const llBetter = metrics.log_loss_rolling_100 < metrics.log_loss_baseline_elo;
  const llDelta = metrics.log_loss_rolling_100 - metrics.log_loss_baseline_elo;

  return (
    <div className={className}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
        <p style={{
          fontSize: '0.63rem', fontWeight: 700, letterSpacing: '0.14em',
          textTransform: 'uppercase', color: '#2E5580',
        }}>
          Evaluación del Modelo — Métricas de Temporada
        </p>
        <span style={{ fontSize: '0.65rem', color: '#374151', fontFamily: 'JetBrains Mono, monospace' }}>
          {metrics.n_games_evaluated} / {metrics.n_games_season} partidos
        </span>
      </div>

      {/* Season progress */}
      <div style={{ height: 4, borderRadius: 2, background: '#1F2937', marginBottom: 14, overflow: 'hidden' }}>
        <div style={{
          height: 4, width: `${completionPct}%`,
          background: 'linear-gradient(90deg, #3B82F6, #60A5FA)', borderRadius: 2,
        }} />
      </div>

      {/* Main 60/40 grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 12 }}>

        {/* ── BRIER SCORE — Primary KPI (audit 2.4: 60% visual weight) ── */}
        <div style={{
          background: '#0D1117',
          border: `2px solid ${brierBetter ? 'rgba(34,197,94,0.35)' : 'rgba(245,158,11,0.35)'}`,
          borderRadius: 12,
          padding: '16px 18px',
          position: 'relative',
          overflow: 'hidden',
        }}>
          {/* PRIMARY KPI badge */}
          <div style={{
            position: 'absolute', top: 10, right: 12,
            fontSize: '0.58rem', fontWeight: 800, letterSpacing: '0.12em',
            padding: '2px 7px', borderRadius: 4,
            background: 'rgba(59,130,246,0.12)', color: '#3B82F6',
            border: '1px solid rgba(59,130,246,0.3)',
          }}>
            KPI PRINCIPAL
          </div>

          {/* Label */}
          <p style={{
            fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.14em',
            textTransform: 'uppercase', color: '#374151', marginBottom: 8,
          }}>
            Brier Score
          </p>

          {/* Period tabs */}
          <div style={{ display: 'flex', gap: 6, marginBottom: 12 }}>
            {(['30d', '90d', 'temporada'] as BrierPeriod[]).map((p) => (
              <PeriodTab
                key={p}
                label={p === 'temporada' ? 'Temporada' : p}
                active={period === p}
                available={brierByPeriod[p] !== undefined}
                onClick={() => brierByPeriod[p] !== undefined && setPeriod(p)}
              />
            ))}
          </div>

          {/* Hero number */}
          <div style={{ display: 'flex', alignItems: 'flex-end', gap: 12, marginBottom: 12 }}>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '2.8rem', fontWeight: 800, lineHeight: 1,
              color: brierBetter ? '#22C55E' : '#F59E0B',
            }}>
              {activeBrier.toFixed(3)}
            </span>
            <div style={{ paddingBottom: 4 }}>
              <span style={{
                display: 'inline-block', fontSize: '0.68rem', fontWeight: 700,
                padding: '3px 9px', borderRadius: 12,
                background: brierBetter ? 'rgba(34,197,94,0.12)' : 'rgba(245,158,11,0.12)',
                color: brierBetter ? '#22C55E' : '#F59E0B',
                border: `1px solid ${brierBetter ? 'rgba(34,197,94,0.3)' : 'rgba(245,158,11,0.3)'}`,
              }}>
                {brierBetter ? '▼ mejor' : '▲ por encima'} benchmark
              </span>
              <p style={{
                fontSize: '0.68rem', color: '#546E7A', marginTop: 4,
                fontFamily: 'JetBrains Mono, monospace',
              }}>
                benchmark (ELO): {benchmark.toFixed(3)}{' '}
                <span style={{ color: brierBetter ? '#22C55E' : '#F59E0B' }}>
                  ({brierDeltaStr})
                </span>
              </p>
            </div>
          </div>

          {/* Visual progress bar */}
          <TrendBar value={activeBrier} benchmark={benchmark} />

          {/* Bar legend */}
          <div style={{
            display: 'flex', justifyContent: 'space-between',
            marginTop: 4, fontSize: '0.6rem', color: '#374151',
          }}>
            <span>← menor = mejor</span>
            <span>benchmark: {benchmark.toFixed(3)}</span>
          </div>

          {/* n context + trend */}
          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 10 }}>
            <span style={{ fontSize: '0.65rem', color: '#374151', fontFamily: 'JetBrains Mono, monospace' }}>
              n = {activeN.toLocaleString()} partidos
              {period !== 'temporada' && brierByPeriod[period] === undefined && (
                <span style={{ color: '#4B5563', marginLeft: 4 }}>(datos insuficientes)</span>
              )}
            </span>
            {trendLabel && (
              <span style={{ fontSize: '0.65rem', fontWeight: 700, color: trendLabel.color }}>
                {trendLabel.text}
              </span>
            )}
          </div>

          {/* Improvement interpretation */}
          <div style={{
            marginTop: 10, padding: '7px 10px', borderRadius: 6,
            background: brierBetter ? 'rgba(34,197,94,0.06)' : 'rgba(245,158,11,0.06)',
            border: `1px solid ${brierBetter ? 'rgba(34,197,94,0.2)' : 'rgba(245,158,11,0.2)'}`,
            fontSize: '0.68rem', lineHeight: 1.5,
            color: brierBetter ? '#4ADE80' : '#FCD34D',
          }}>
            {brierBetter
              ? `▶ El modelo supera al baseline ELO en ${Math.abs(brierDelta * 1000).toFixed(1)} miliBrier. Calibración estadísticamente útil.`
              : `▶ Brier por encima del baseline ELO (${Math.abs(brierDelta * 1000).toFixed(1)} mB). Revisar calibración de probabilidades de victoria.`
            }
          </div>
        </div>

        {/* ── LOG-LOSS — Secondary metric ── */}
        <div style={{
          background: '#0D1117', border: '1px solid #1F2937',
          borderRadius: 12, padding: '16px 18px',
          display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <p style={{
            fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.14em',
            textTransform: 'uppercase', color: '#374151',
          }}>
            Log-Loss (últimos 100)
          </p>

          <span style={{
            fontFamily: 'JetBrains Mono, monospace',
            fontSize: '2.0rem', fontWeight: 800, lineHeight: 1,
            color: llBetter ? '#22C55E' : '#F59E0B',
          }}>
            {metrics.log_loss_rolling_100.toFixed(3)}
          </span>

          <div>
            <span style={{
              fontSize: '0.65rem', fontWeight: 700, padding: '2px 8px', borderRadius: 10,
              background: llBetter ? 'rgba(34,197,94,0.1)' : 'rgba(245,158,11,0.1)',
              color: llBetter ? '#22C55E' : '#F59E0B',
              border: `1px solid ${llBetter ? 'rgba(34,197,94,0.25)' : 'rgba(245,158,11,0.25)'}`,
            }}>
              {llBetter ? '▼ mejor' : '▲ por encima'} baseline
            </span>
            <p style={{
              fontSize: '0.68rem', color: '#546E7A', marginTop: 6,
              fontFamily: 'JetBrains Mono, monospace',
            }}>
              Baseline Elo: {metrics.log_loss_baseline_elo.toFixed(3)}{' '}
              <span style={{ color: llBetter ? '#22C55E' : '#F59E0B' }}>
                ({llDelta >= 0 ? '+' : ''}{llDelta.toFixed(3)})
              </span>
            </p>
            <p style={{ fontSize: '0.63rem', color: '#374151', marginTop: 4 }}>
              n = {Math.min(metrics.n_games_evaluated, 100).toLocaleString()} partidos
            </p>
          </div>

          {/* Bar */}
          <div style={{ position: 'relative', height: 6, borderRadius: 3, background: '#1F2937' }}>
            <div style={{
              position: 'absolute', left: 0, top: 0, height: 6,
              width: `${Math.min(98, (metrics.log_loss_rolling_100 / (metrics.log_loss_baseline_elo * 1.3)) * 100)}%`,
              background: llBetter ? '#22C55E' : '#F59E0B', borderRadius: 3, opacity: 0.7,
            }} />
          </div>

          {/* Accuracy if available */}
          {metrics.accuracy_season !== undefined && (
            <div style={{
              padding: '8px 10px', borderRadius: 6,
              background: '#111827', border: '1px solid #1F2937',
            }}>
              <p style={{ fontSize: '0.6rem', color: '#374151', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 3 }}>
                Accuracy (ganador)
              </p>
              <span style={{
                fontSize: '1.2rem', fontWeight: 700,
                fontFamily: 'JetBrains Mono, monospace',
                color: metrics.accuracy_season >= 0.6 ? '#22C55E' : metrics.accuracy_season >= 0.5 ? '#F59E0B' : '#EF4444',
              }}>
                {(metrics.accuracy_season * 100).toFixed(1)}%
              </span>
              {metrics.accuracy_30d !== undefined && (
                <span style={{ fontSize: '0.65rem', color: '#546E7A', marginLeft: 6, fontFamily: 'JetBrains Mono, monospace' }}>
                  30d: {(metrics.accuracy_30d * 100).toFixed(1)}%
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Percentile rank badge */}
      {metrics.model_percentile !== undefined && (
        <div style={{
          marginTop: 12, padding: '8px 14px', borderRadius: 8,
          background: '#0D1117', border: '1px solid #1F2937',
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <span style={{
            fontSize: '0.63rem', fontWeight: 700, letterSpacing: '0.1em',
            textTransform: 'uppercase', color: '#374151',
          }}>
            Percentil de temporada
          </span>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontWeight: 800, fontSize: '1.1rem',
            color: metrics.model_percentile >= 75 ? '#22C55E' : metrics.model_percentile >= 50 ? '#F59E0B' : '#EF4444',
          }}>
            {metrics.model_percentile}°
          </span>
          <span style={{ fontSize: '0.72rem', color: '#546E7A' }}>
            {metrics.model_percentile >= 75
              ? '— Modelo en top-25% de la temporada'
              : metrics.model_percentile >= 50
              ? '— Por encima de la media'
              : '— Por debajo de la media'}
          </span>
        </div>
      )}

      {/* Per-game detail — collapsed (Bug 5: never primary) */}
      {gameLogLoss !== undefined && (
        <div style={{ marginTop: 10 }}>
          <button
            onClick={() => setShowGameDetail((v) => !v)}
            style={{
              background: 'none', border: 'none', color: '#374151',
              fontSize: '0.7rem', cursor: 'pointer', padding: '4px 0', letterSpacing: '0.05em',
            }}
          >
            {showGameDetail ? '▲' : '▼'} Detalle de este partido
          </button>
          {showGameDetail && (
            <div style={{
              marginTop: 6, padding: '10px 14px', borderRadius: 8,
              background: '#0D1117', border: '1px solid #1F2937',
              fontSize: '0.75rem', color: '#546E7A',
              fontFamily: 'JetBrains Mono, monospace',
            }}>
              <p>
                Log-Loss (partido individual):{' '}
                <span style={{ color: '#78909C', fontWeight: 700 }}>{gameLogLoss.toFixed(3)}</span>
                <span style={{
                  marginLeft: 8, fontSize: '0.68rem', color: '#374151',
                  fontFamily: 'sans-serif', fontStyle: 'italic',
                }}>
                  (n=1, sin significancia estadística — solo referencial)
                </span>
              </p>
              <p style={{ marginTop: 4 }}>
                Contribución al Log-Loss acumulado:{' '}
                <span style={{ color: '#78909C' }}>
                  +{(gameLogLoss / Math.max(metrics.n_games_evaluated, 1)).toFixed(4)}
                </span>{' '}
                <span style={{ fontSize: '0.68rem', fontFamily: 'sans-serif' }}>(marginal)</span>
              </p>
            </div>
          )}
        </div>
      )}

      <p style={{ marginTop: 12, fontSize: '0.65rem', color: '#374151', fontStyle: 'italic' }}>
        * Brier Score: métrica primaria de calibración probabilística. Rango [0,1], menor = mejor.
        Benchmark = 0.228 (MLB Elo baseline). La calificación se muestra solo para n ≥ 30.
      </p>
    </div>
  );
}
