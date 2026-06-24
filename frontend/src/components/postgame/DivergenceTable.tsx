/**
 * DivergenceTable — Unified model-vs-manager comparison (Bug 3 + 4 fix).
 * Roadmap 1.2: IC90 display — only color green/red when interval excludes zero.
 * Roadmap 1.3: Filter to significant divergences by default; add methodology note.
 *
 * Table: Slot | Modelo sugería | Manager eligió | ΔE[R] ± IC90 | Veredicto
 */
import { useState } from 'react';
import type { DivergenceRow, FeatureFactor } from '../../types';

interface DivergenceTableProps {
  divergences: DivergenceRow[];
  /** Actual-game results keyed by slot (optional, shown in Manager column). */
  results?: Record<number, string>;
  /**
   * audit F06: cuando los datos provienen del modo demo, los factores son
   * ILUSTRATIVOS (no SHAP real del modelo). Muestra un aviso visible para que
   * el usuario no los confunda con explicaciones reales.
   */
  isDemo?: boolean;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Returns colors only when statistically significant; otherwise gray. */
function deltaColor(delta: number, match: boolean, significant?: boolean): string {
  if (match) return '#6B7280';
  if (!significant) return '#4B5563'; // within variance — gray
  if (delta > 0.005) return '#22C55E';  // manager better — green
  if (delta < -0.005) return '#EF4444'; // model was better — red
  return '#6B7280';
}

function verdictLabel(
  delta: number,
  match: boolean,
  significant?: boolean
): { text: string; color: string; shape: string } {
  if (match) return { text: 'Coincide', color: '#6B7280', shape: '=' };
  if (!significant) return { text: 'Dentro de varianza', color: '#4B5563', shape: '~' };
  if (delta > 0.005) return { text: 'Manager mejor', color: '#22C55E', shape: '✓' };
  if (delta < -0.005) return { text: 'Coste al equipo', color: '#EF4444', shape: '✗' };
  return { text: 'Sin impacto', color: '#6B7280', shape: '≈' };
}

function deltaArrow(delta: number, match: boolean, significant?: boolean): string {
  if (match) return '=';
  if (!significant) return '~';
  if (delta > 0.005) return '↑';
  if (delta < -0.005) return '↓';
  return '≈';
}

// ── Factor tooltip (Bug 4 — feature importance) ───────────────────────────────

function FactorPanel({ factors, isDemo }: { factors: FeatureFactor[]; isDemo?: boolean }) {
  if (!factors.length) return null;

  return (
    <tr>
      <td colSpan={5} style={{ background: '#0A1220', padding: 0 }}>
        <div
          style={{
            borderLeft: '3px solid #3B82F6',
            padding: '12px 16px',
            marginLeft: 8,
            marginBottom: 4,
          }}
        >
          <p
            style={{
              fontSize: '0.68rem',
              fontWeight: 700,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: '#3B82F6',
              marginBottom: 8,
            }}
          >
            Factores que explican esta decisión
          </p>
          {isDemo && (
            <p
              style={{
                fontSize: '0.66rem',
                color: '#FCD34D',
                background: 'rgba(245,158,11,0.10)',
                border: '1px solid rgba(245,158,11,0.35)',
                borderRadius: 5,
                padding: '4px 8px',
                marginBottom: 8,
              }}
            >
              ⚠ Datos de demostración — factores ilustrativos, NO SHAP real del modelo.
            </p>
          )}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {factors.map((f, i) => {
              const fc =
                f.direction === 'favor_modelo'
                  ? '#60A5FA'
                  : '#F97316';
              const icon = f.direction === 'favor_modelo' ? '🤖' : '👨‍💼';
              return (
                <div
                  key={i}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 8,
                    fontSize: '0.75rem',
                  }}
                >
                  <span style={{ fontSize: '0.7rem', minWidth: 20 }}>{icon}</span>
                  <span style={{ color: '#90A4AE', lineHeight: 1.5 }}>
                    {f.factor}
                  </span>
                  <span
                    style={{
                      marginLeft: 'auto',
                      fontFamily: 'JetBrains Mono, monospace',
                      fontSize: '0.7rem',
                      color: fc,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    EV {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
                  </span>
                </div>
              );
            })}
          </div>
          <p
            style={{
              fontSize: '0.65rem',
              color: '#374151',
              marginTop: 8,
            }}
          >
            🤖 = favorece al modelo &nbsp;·&nbsp; 👨‍💼 = favorece al manager &nbsp;·&nbsp;
            EV ≈ contribución estimada en carreras esperadas
          </p>
        </div>
      </td>
    </tr>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

export default function DivergenceTable({
  divergences,
  results = {},
  isDemo = false,
}: DivergenceTableProps) {
  const [expandedSlot, setExpandedSlot] = useState<number | null>(null);
  // Roadmap 1.3: show significant-only by default to reduce noise
  const [showOnlySignificant, setShowOnlySignificant] = useState(true);

  // A row is "significant" if significant===true, or if ic90 is absent and |delta|>0.01 (legacy)
  const isRowSignificant = (d: DivergenceRow) =>
    d.significant === true || (d.significant === undefined && Math.abs(d.delta_er) > 0.01);

  const rows = showOnlySignificant
    ? divergences.filter((d) => d.match || isRowSignificant(d))
    : divergences;

  // ── Totals ───────────────────────────────────────────────────────────────
  const totalDelta = divergences.reduce(
    (acc, d) => acc + (d.match ? 0 : d.delta_er),
    0
  );
  const nDivergences = divergences.filter((d) => !d.match).length;
  const nSignificant = divergences.filter((d) => !d.match && isRowSignificant(d)).length;
  const positiveTotal = totalDelta > 0;

  return (
    <div>
      {/* Controls */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: 8,
          flexWrap: 'wrap',
          gap: 8,
        }}
      >
        <p
          style={{
            fontSize: '0.63rem',
            fontWeight: 700,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            color: '#2E5580',
          }}
        >
          {nDivergences} divergencia(s) · <span style={{ color: nSignificant > 0 ? '#F59E0B' : '#4B5563' }}>
            {nSignificant} significativa(s)
          </span>
        </p>
        <button
          onClick={() => setShowOnlySignificant((v) => !v)}
          style={{
            fontSize: '0.7rem',
            padding: '3px 10px',
            borderRadius: 6,
            border: `1px solid ${showOnlySignificant ? '#F59E0B' : '#1F2937'}`,
            background: showOnlySignificant ? 'rgba(245,158,11,0.10)' : 'transparent',
            color: showOnlySignificant ? '#FCD34D' : '#6B7280',
            cursor: 'pointer',
          }}
        >
          {showOnlySignificant ? '◈ Mostrando solo significativas' : '▸ Mostrar todo'}
        </button>
      </div>

      {/* Roadmap 1.3 — methodology note */}
      <div
        style={{
          marginBottom: 12,
          padding: '8px 12px',
          borderRadius: 6,
          background: 'rgba(30,56,80,0.4)',
          border: '1px solid #1E3850',
          fontSize: '0.72rem',
          color: '#4B6580',
          lineHeight: 1.6,
        }}
      >
        ℹ &nbsp;El orden del lineup representa <strong style={{ color: '#546E7A' }}>~1–5 carreras/temporada</strong>{' '}
        (~0.02 E[R]/decisión). Solo se resaltan divergencias cuyo IC90 excluye el cero.{' '}
        Filas en gris = dentro de la varianza estadística del modelo.
      </div>

      {/* Table */}
      <div style={{ overflowX: 'auto' }}>
        <table
          style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}
        >
          <thead>
            <tr style={{ background: '#0D1117', borderBottom: '1px solid #1F2937' }}>
              {['Slot', 'Modelo sugería', 'Manager eligió', 'ΔE[R] ± IC90', 'Veredicto'].map(
                (h) => (
                  <th
                    key={h}
                    style={{
                      padding: '8px 10px',
                      textAlign: h === 'Slot' || h === 'ΔE[R] ± IC90' ? 'center' : 'left',
                      fontSize: '0.63rem',
                      fontWeight: 700,
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      color: '#374151',
                    }}
                  >
                    {h}
                  </th>
                )
              )}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const significant = isRowSignificant(row);
              const dc = deltaColor(row.delta_er, row.match, significant);
              const verdict = verdictLabel(row.delta_er, row.match, significant);
              const arrow = deltaArrow(row.delta_er, row.match, significant);
              const isExpanded = expandedSlot === row.slot;
              const canExpand = !row.match && row.top_factors.length > 0;
              const gameResult = results[row.slot];

              return (
                <>
                  <tr
                    key={row.slot}
                    onClick={() =>
                      canExpand &&
                      setExpandedSlot(isExpanded ? null : row.slot)
                    }
                    style={{
                      background: isExpanded
                        ? '#0E1830'
                        : row.match
                        ? '#111827'
                        : significant
                        ? '#0F1929'
                        : '#0D1117',
                      borderBottom: '1px solid #1F2937',
                      cursor: canExpand ? 'pointer' : 'default',
                      transition: 'background 0.15s',
                      opacity: row.match ? 0.7 : 1,
                    }}
                    onMouseEnter={(e) => {
                      if (!isExpanded)
                        (e.currentTarget as HTMLTableRowElement).style.background = '#1a2035';
                    }}
                    onMouseLeave={(e) => {
                      if (!isExpanded)
                        (e.currentTarget as HTMLTableRowElement).style.background =
                          row.match ? '#111827' : significant ? '#0F1929' : '#0D1117';
                    }}
                  >
                    {/* Slot */}
                    <td style={{ padding: '9px 10px', textAlign: 'center' }}>
                      <span
                        style={{
                          display: 'inline-flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          width: 24,
                          height: 24,
                          borderRadius: 6,
                          background: '#1F2937',
                          fontFamily: 'JetBrains Mono, monospace',
                          fontWeight: 700,
                          fontSize: '0.75rem',
                          color: '#9CA3AF',
                        }}
                      >
                        {row.slot}
                      </span>
                    </td>

                    {/* Modelo sugería */}
                    <td style={{ padding: '9px 10px' }}>
                      <div>
                        <span style={{ color: '#90CAF9', fontWeight: 600 }}>
                          {row.model_player}
                        </span>
                        <span style={{ color: '#374151', fontSize: '0.72rem', marginLeft: 6 }}>
                          {row.model_pos}
                        </span>
                      </div>
                    </td>

                    {/* Manager eligió */}
                    <td style={{ padding: '9px 10px' }}>
                      <div>
                        <span
                          style={{
                            color: row.match ? '#90CAF9' : significant ? '#FFC107' : '#6B7280',
                            fontWeight: row.match ? 400 : significant ? 600 : 400,
                          }}
                        >
                          {row.manager_player}
                        </span>
                        <span style={{ color: '#374151', fontSize: '0.72rem', marginLeft: 6 }}>
                          {row.manager_pos}
                        </span>
                        {gameResult && (
                          <span
                            style={{
                              display: 'block',
                              fontSize: '0.68rem',
                              color: '#546E7A',
                              fontFamily: 'JetBrains Mono, monospace',
                              marginTop: 2,
                            }}
                          >
                            {gameResult}
                          </span>
                        )}
                      </div>
                    </td>

                    {/* ΔE[R] ± IC90 (Roadmap 1.2) */}
                    <td style={{ padding: '9px 10px', textAlign: 'center' }}>
                      {row.match ? (
                        <span style={{ fontFamily: 'JetBrains Mono, monospace', color: '#374151', fontSize: '0.78rem' }}>
                          —
                        </span>
                      ) : (
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 1 }}>
                          <span
                            style={{
                              fontFamily: 'JetBrains Mono, monospace',
                              fontWeight: 700,
                              fontSize: '0.82rem',
                              color: dc,
                            }}
                          >
                            {row.delta_er >= 0 ? '+' : ''}{row.delta_er.toFixed(3)} {arrow}
                          </span>
                          {row.ic90_halfwidth !== undefined && (
                            <span style={{
                              fontFamily: 'JetBrains Mono, monospace',
                              fontSize: '0.62rem',
                              color: '#374151',
                            }}>
                              ± {row.ic90_halfwidth.toFixed(3)}
                            </span>
                          )}
                        </div>
                      )}
                    </td>

                    {/* Veredicto */}
                    <td style={{ padding: '9px 10px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                        <span
                          style={{
                            fontSize: '0.7rem',
                            fontWeight: 700,
                            color: verdict.color,
                            fontFamily: 'JetBrains Mono, monospace',
                          }}
                        >
                          {verdict.shape}
                        </span>
                        <span style={{ fontSize: '0.73rem', color: significant ? '#78909C' : '#4B5563' }}>
                          {verdict.text}
                        </span>
                        {canExpand && significant && (
                          <span
                            style={{
                              marginLeft: 'auto',
                              fontSize: '0.65rem',
                              color: '#3B82F6',
                              opacity: 0.7,
                            }}
                          >
                            {isExpanded ? '▲' : '▼'} Por qué
                          </span>
                        )}
                      </div>
                    </td>
                  </tr>

                  {/* Expandable factor panel (Bug 4) */}
                  {isExpanded && <FactorPanel factors={row.top_factors} isDemo={isDemo} />}
                </>
              );
            })}

            {/* Total row */}
            <tr
              style={{
                background: '#0D1117',
                borderTop: '2px solid #1F2937',
              }}
            >
              <td colSpan={3} style={{ padding: '10px 10px' }}>
                <span
                  style={{
                    fontSize: '0.72rem',
                    color: '#546E7A',
                    fontStyle: 'italic',
                  }}
                >
                  {nDivergences === 0
                    ? 'El manager siguió exactamente el lineup del modelo.'
                    : `${nDivergences} divergencia(s) · ${nSignificant} superan el IC90 (estadísticamente significativas).`}
                </span>
              </td>
              <td style={{ padding: '10px 10px', textAlign: 'center' }}>
                <span
                  style={{
                    fontFamily: 'JetBrains Mono, monospace',
                    fontWeight: 700,
                    fontSize: '0.88rem',
                    color: positiveTotal ? '#22C55E' : totalDelta < 0 ? '#EF4444' : '#6B7280',
                  }}
                >
                  {totalDelta >= 0 ? '+' : ''}{totalDelta.toFixed(3)}
                </span>
              </td>
              <td style={{ padding: '10px 10px' }}>
                <span
                  style={{
                    fontSize: '0.7rem',
                    color: '#546E7A',
                    fontFamily: 'JetBrains Mono, monospace',
                  }}
                >
                  ΔE[R] neto
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Summary narrative */}
      {nSignificant > 0 && (
        <div
          style={{
            marginTop: 10,
            padding: '10px 14px',
            borderRadius: 8,
            borderLeft: `3px solid ${positiveTotal ? '#22C55E' : '#EF4444'}`,
            background: positiveTotal
              ? 'rgba(34,197,94,0.06)'
              : 'rgba(239,68,68,0.06)',
            fontSize: '0.78rem',
            color: '#90A4AE',
            lineHeight: 1.6,
          }}
        >
          <strong style={{ color: '#E8EAF6' }}>{nSignificant}</strong>{' '}
          decisión(es) superaron el umbral estadístico (IC90), con impacto neto de{' '}
          <strong
            style={{
              color: positiveTotal ? '#22C55E' : '#EF4444',
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            {totalDelta >= 0 ? '+' : ''}{totalDelta.toFixed(3)} E[R]
          </strong>{' '}
          vs. el lineup óptimo sugerido.{' '}
          <em style={{ color: '#546E7A', fontSize: '0.72rem' }}>
            Las {nDivergences - nSignificant} restante(s) están dentro del ruido estadístico.
          </em>
        </div>
      )}
    </div>
  );
}
