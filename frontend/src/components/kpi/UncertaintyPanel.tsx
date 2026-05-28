/**
 * UncertaintyPanel — Roadmap 0.5 (replaces ConfidencePanel)
 *
 * Replaces the vague "Confianza 82%" with a properly defined metric:
 *   - Primary display: P10–P90 interval (prediction range)
 *   - Secondary: uncertainty_level badge (low/medium/high)
 *   - No hardcoded percentages without operational definition
 *
 * A statistician can explain every number shown here.
 */
import { useState } from 'react';
import type { ModelConfidenceDetail } from '../../types';

interface UncertaintyPanelProps {
  /** P10 expected runs (10th percentile of simulation). */
  p10?: number;
  /** P50 expected runs (median). */
  p50?: number;
  /** P90 expected runs (90th percentile). */
  p90?: number;
  /** E[R] point estimate. */
  er?: number;
  /** Uncertainty classification from the API. */
  uncertaintyLevel?: 'low' | 'medium' | 'high';
  /** Monte Carlo stability sigma. */
  stdDev?: number;
  /** Win probability confidence interval. */
  winProbCiLow?: number;
  winProbCiHigh?: number;
  /** Legacy detail for methodology tooltip. */
  detail?: ModelConfidenceDetail;
}

const LEVEL_CONFIG = {
  low: {
    color: '#00FF87',
    label: 'Baja incertidumbre',
    bg: 'rgba(0,255,135,0.08)',
    /** Operational protocol (audit 3.4) */
    action: 'Confianza estructural suficiente. Proceder con lineup óptimo.',
    action_color: '#00FF87',
  },
  medium: {
    color: '#FFB800',
    label: 'Incertidumbre moderada',
    bg: 'rgba(255,184,0,0.08)',
    action: 'Usar predicción con reserva. Validar con intel adicional antes de decidir.',
    action_color: '#FFB800',
  },
  high: {
    color: '#EF4444',
    label: 'Alta incertidumbre',
    bg: 'rgba(239,68,68,0.08)',
    action: 'Aumentar umbral de confianza mínima al 60% antes de actuar. Revisar datos del pitcher y condiciones del parque.',
    action_color: '#EF4444',
  },
};

export default function UncertaintyPanel({
  p10, p50, p90, er, uncertaintyLevel = 'medium',
  stdDev, winProbCiLow, winProbCiHigh, detail,
}: UncertaintyPanelProps) {
  const [open, setOpen] = useState(false);

  const lvl = LEVEL_CONFIG[uncertaintyLevel] ?? LEVEL_CONFIG.medium;
  const intervalWidth = (p10 !== undefined && p90 !== undefined)
    ? p90 - p10 : (stdDev !== undefined ? stdDev * 2.56 : undefined);  // 2.56 ≈ P10–P90 multiplier for normal

  const erDisplay = p50 ?? er;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, position: 'relative' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <p style={{
          fontSize: '0.63rem', fontWeight: 700, letterSpacing: '0.14em',
          textTransform: 'uppercase', color: '#546E7A',
        }}>
          Intervalo de Predicción
        </p>
        <span style={{
          fontSize: '0.68rem', fontWeight: 700, padding: '2px 8px',
          borderRadius: 4, background: lvl.bg, color: lvl.color,
          border: `1px solid ${lvl.color}40`,
        }}>
          {uncertaintyLevel?.toUpperCase() ?? 'MEDIUM'}
        </span>
      </div>

      {/* P10–P50–P90 display */}
      {p10 !== undefined && p90 !== undefined ? (
        <div>
          {/* Interval bar */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4,
          }}>
            {/* P10 */}
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '0.58rem', color: '#374151', marginBottom: 1 }}>P10</p>
              <span style={{
                fontFamily: 'JetBrains Mono, monospace', fontSize: '1.0rem',
                fontWeight: 700, color: '#546E7A',
              }}>
                {p10.toFixed(1)}
              </span>
            </div>

            {/* Visual bar */}
            <div style={{ flex: 1, position: 'relative', height: 10 }}>
              <div style={{
                position: 'absolute', inset: 0, background: '#0D1117',
                borderRadius: 4,
              }} />
              {/* IQR fill (P10-P90 span) */}
              <div style={{
                position: 'absolute', inset: 0,
                background: `linear-gradient(90deg, ${lvl.color}20, ${lvl.color}50, ${lvl.color}20)`,
                borderRadius: 4,
              }} />
              {/* P50 line */}
              {p50 !== undefined && (
                <div style={{
                  position: 'absolute',
                  left: `${Math.min(100, Math.max(0, ((p50 - p10) / (p90 - p10)) * 100))}%`,
                  top: '-2px', bottom: '-2px', width: 2,
                  background: lvl.color, borderRadius: 1,
                }} title={`P50: ${p50.toFixed(2)}`} />
              )}
            </div>

            {/* P90 */}
            <div style={{ textAlign: 'center' }}>
              <p style={{ fontSize: '0.58rem', color: '#374151', marginBottom: 1 }}>P90</p>
              <span style={{
                fontFamily: 'JetBrains Mono, monospace', fontSize: '1.0rem',
                fontWeight: 700, color: '#546E7A',
              }}>
                {p90.toFixed(1)}
              </span>
            </div>
          </div>

          {/* E[R] ± summary */}
          {erDisplay !== undefined && intervalWidth !== undefined && (
            <p style={{
              fontSize: '0.75rem', fontFamily: 'JetBrains Mono, monospace',
              color: '#78909C', marginTop: 4,
            }}>
              E[R] = {erDisplay.toFixed(2)}{' '}
              <span style={{ color: '#374151' }}>
                ± {(intervalWidth / 2).toFixed(2)} (P10–P90)
              </span>
            </p>
          )}
        </div>
      ) : (
        /* Fallback: just show E[R] with σ */
        <div>
          {erDisplay !== undefined && (
            <p style={{
              fontFamily: 'JetBrains Mono, monospace', fontSize: '1.1rem',
              fontWeight: 700, color: lvl.color,
            }}>
              {erDisplay.toFixed(2)}
              {stdDev !== undefined && (
                <span style={{ fontSize: '0.75rem', color: '#546E7A', fontWeight: 400 }}>
                  {' '}± {stdDev.toFixed(2)}σ
                </span>
              )}
            </p>
          )}
        </div>
      )}

      {/* Win Prob CI */}
      {winProbCiLow !== undefined && winProbCiHigh !== undefined && (
        <div style={{
          padding: '6px 10px', borderRadius: 6,
          background: '#0D1117', border: '1px solid #1F2937',
          fontSize: '0.72rem', fontFamily: 'JetBrains Mono, monospace',
          color: '#78909C',
        }}>
          P(W) IC90:{' '}
          <span style={{ color: '#90CAF9' }}>
            {(winProbCiLow * 100).toFixed(0)}% – {(winProbCiHigh * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Actionable protocol (audit 3.4 — HIGH must not be decorative) */}
      <div style={{
        padding: '7px 10px', borderRadius: 6,
        background: `${lvl.color}10`,
        border: `1px solid ${lvl.color}30`,
        fontSize: '0.68rem',
        color: lvl.action_color,
        lineHeight: 1.5,
      }}>
        <span style={{ fontWeight: 700, marginRight: 4 }}>▶</span>
        {lvl.action}
      </div>

      {/* Footer with methodology link */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <p style={{ fontSize: '0.63rem', color: '#374151' }}>
          {lvl.label}
          {stdDev !== undefined && ` · σ = ${stdDev.toFixed(2)}`}
        </p>
        <button
          style={{
            fontSize: '0.62rem', color: '#374151',
            background: 'none', border: 'none', cursor: 'pointer',
            textDecoration: 'underline', textDecorationStyle: 'dotted',
          }}
          onMouseEnter={() => setOpen(true)}
          onMouseLeave={() => setOpen(false)}
          onClick={() => setOpen(o => !o)}
        >
          ℹ Metodología
        </button>
      </div>

      {/* Methodology tooltip */}
      {open && (
        <div style={{
          position: 'absolute', right: 0, top: 'calc(100% + 4px)',
          background: '#0D1117', border: '1px solid #374151',
          borderRadius: 8, padding: '12px 14px',
          width: 280, zIndex: 40,
          fontSize: '0.72rem', fontFamily: 'JetBrains Mono, monospace',
          color: '#9CA3AF', boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
          lineHeight: 1.7,
        }}>
          <p style={{ color: '#E8EAF6', fontWeight: 700, marginBottom: 8 }}>
            Intervalo de predicción — Definición
          </p>
          <ul style={{ paddingLeft: 0, listStyle: 'none' }}>
            <li>• <strong style={{ color: '#E8EAF6' }}>P10–P90:</strong> Rango de carreras que el modelo cubre con 80% de probabilidad (simulación Monte Carlo).</li>
            <li>• <strong style={{ color: '#E8EAF6' }}>Baja incertidumbre:</strong> Intervalo {'<'} 4 carreras.</li>
            <li>• <strong style={{ color: '#E8EAF6' }}>Alta incertidumbre:</strong> Intervalo {'>'} 7 carreras.</li>
            <li>• <strong style={{ color: '#E8EAF6' }}>P(W) IC90:</strong> Intervalo de confianza de la probabilidad de victoria.</li>
          </ul>
          {detail && (
            <>
              <p style={{ borderTop: '1px solid #1F2937', paddingTop: 8, marginTop: 8, color: '#546E7A' }}>
                σ MC = {detail.components.montecarlo_stability_sigma?.toFixed(3) ?? '—'} runs
              </p>
              {detail.components.backtesting_log_loss !== null && detail.components.backtesting_log_loss !== undefined ? (
                <p style={{ color: '#546E7A' }}>
                  Log-Loss backtest: {detail.components.backtesting_log_loss.toFixed(4)}
                  {detail.components.backtesting_sample_n !== null && detail.components.backtesting_sample_n !== undefined
                    ? ` (n=${detail.components.backtesting_sample_n})`
                    : ''}
                </p>
              ) : (
                <p style={{ color: '#374151', fontSize: '0.65rem' }}>
                  Ejecuta backtest.py para métricas verificables
                </p>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
