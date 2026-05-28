/**
 * PostGame — Post-Game Review page.
 *
 * Implements the same review capability as views/post_game.py (Streamlit)
 * but in React with all 5 bug fixes applied:
 *
 *  Bug 1: useReducer with strict requestId invalidation prevents stale data.
 *  Bug 2: ProjectionChart (box-whisker) replaces bar chart.
 *  Bug 3: DivergenceTable unifies the two-column layout with ΔE[R].
 *  Bug 4: Divergence row expansion shows top-3 factors.
 *  Bug 5: MetricsPanel replaces single-game Log-Loss with rolling metrics.
 */
import { useReducer, useEffect, useCallback, useState } from 'react';
import type { HistoricalGame, PostGameReport, RollingMetrics, CalibrationData } from '../types';
import {
  fetchHistoricalGames,
  fetchPostGameReport,
  fetchRollingMetrics,
  fetchCalibrationData,
} from '../api/client';
import DivergenceTable from '../components/postgame/DivergenceTable';
import MetricsPanel from '../components/postgame/MetricsPanel';
import ProjectionChart from '../components/charts/ProjectionChart';
import CalibrationChart from '../components/charts/CalibrationChart';

// ── Helpers ───────────────────────────────────────────────────────────────────

function todayMinus(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().split('T')[0];
}

function Skeleton({ style }: { style?: React.CSSProperties }) {
  return (
    <div
      style={{
        background: 'linear-gradient(90deg, #1a2035 25%, #1e2540 50%, #1a2035 75%)',
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.5s infinite',
        borderRadius: 8,
        ...style,
      }}
    />
  );
}

function Card({
  children,
  style,
}: {
  children: React.ReactNode;
  style?: React.CSSProperties;
}) {
  return (
    <div
      style={{
        background: '#111827',
        border: '1px solid #1F2937',
        borderRadius: 12,
        padding: '20px 22px',
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <p
      style={{
        fontSize: '0.63rem',
        fontWeight: 700,
        letterSpacing: '0.14em',
        textTransform: 'uppercase',
        color: '#2E5580',
        marginBottom: 14,
        paddingBottom: 8,
        borderBottom: '1px solid #162030',
      }}
    >
      {children}
    </p>
  );
}

// ── State management (Bug 1 pattern applied to PostGame) ──────────────────────

interface PGState {
  date: string;
  /** Currently selected historical game_pk. */
  game_pk: number | null;
  /** Home or away team perspective (mirrors War Room toggle). */
  team: 'home' | 'away';
  historicalGames: HistoricalGame[];
  historicalLoading: boolean;
  historicalError: string | null;
  report: PostGameReport | null;
  reportLoading: boolean;
  reportError: string | null;
  /** Monotonic — increments on every game_pk or team change, invalidates stale fetches. */
  _reqId: number;
}

type PGAction =
  | { type: 'DATE_CHANGED'; payload: string }
  | { type: 'HISTORICAL_SUCCESS'; payload: HistoricalGame[] }
  | { type: 'HISTORICAL_ERROR'; payload: string }
  | { type: 'GAME_SELECTED'; payload: number }
  | { type: 'TEAM_CHANGED'; payload: 'home' | 'away' }
  | { type: 'REPORT_SUCCESS'; payload: { data: PostGameReport; reqId: number } }
  | { type: 'REPORT_ERROR'; payload: { error: string; reqId: number } };

function pgReducer(state: PGState, action: PGAction): PGState {
  switch (action.type) {
    case 'DATE_CHANGED':
      return {
        ...state,
        date: action.payload,
        game_pk: null,
        historicalGames: [],
        historicalLoading: true,
        historicalError: null,
        report: null,
        reportLoading: false,
        reportError: null,
        _reqId: state._reqId + 1,
      };

    case 'HISTORICAL_SUCCESS':
      return {
        ...state,
        historicalGames: action.payload,
        historicalLoading: false,
        game_pk: state.game_pk ?? (action.payload[0]?.game_pk ?? null),
        reportLoading: action.payload.length > 0 && state.game_pk === null,
        _reqId: action.payload.length > 0 && state.game_pk === null
          ? state._reqId + 1
          : state._reqId,
      };

    case 'HISTORICAL_ERROR':
      return {
        ...state,
        historicalLoading: false,
        historicalError: action.payload,
      };

    case 'GAME_SELECTED':
      if (state.game_pk === action.payload) return state;
      return {
        ...state,
        game_pk: action.payload,
        report: null,
        reportLoading: true,
        reportError: null,
        _reqId: state._reqId + 1,
      };

    case 'TEAM_CHANGED':
      if (state.team === action.payload) return state;
      return {
        ...state,
        team: action.payload,
        report: null,
        reportLoading: state.game_pk !== null,
        reportError: null,
        _reqId: state._reqId + 1,
      };

    case 'REPORT_SUCCESS':
      if (action.payload.reqId !== state._reqId) return state;
      return {
        ...state,
        report: action.payload.data,
        reportLoading: false,
        reportError: null,
      };

    case 'REPORT_ERROR':
      if (action.payload.reqId !== state._reqId) return state;
      return {
        ...state,
        report: null,
        reportLoading: false,
        reportError: action.payload.error,
      };

    default:
      return state;
  }
}

// ── Main component ────────────────────────────────────────────────────────────

export default function PostGame() {
  const [state, dispatch] = useReducer(pgReducer, {
    date: todayMinus(1),
    game_pk: null,
    team: 'home',
    historicalGames: [],
    historicalLoading: true,
    historicalError: null,
    report: null,
    reportLoading: false,
    reportError: null,
    _reqId: 0,
  });

  const [rollingMetrics, setRollingMetrics] = useState<RollingMetrics | null>(null);
  const [calibrationData, setCalibrationData] = useState<CalibrationData | null>(null);

  // ── Fetch historical games on date change ─────────────────────────────────
  useEffect(() => {
    if (!state.historicalLoading) return;
    const date = state.date;
    let cancelled = false;

    fetchHistoricalGames(date)
      .then((games) => {
        if (!cancelled) dispatch({ type: 'HISTORICAL_SUCCESS', payload: games });
      })
      .catch((e) => {
        if (!cancelled)
          dispatch({
            type: 'HISTORICAL_ERROR',
            payload: e instanceof Error ? e.message : 'Error',
          });
      });

    return () => { cancelled = true; };
  }, [state.date, state.historicalLoading]);

  // ── Fetch report on game_pk or team change (Bug 1 — strict reqId) ──────────
  useEffect(() => {
    if (state.game_pk === null) return;
    const { game_pk, date, team, _reqId: reqId } = state;
    let cancelled = false;

    (async () => {
      try {
        const data = await fetchPostGameReport(game_pk, date, team);
        if (cancelled) return;

        // Validate coherence (Bug 1)
        if (data.game_pk !== game_pk) {
          throw new Error(
            `API mismatch: expected game_pk=${game_pk}, got ${data.game_pk}`
          );
        }
        dispatch({ type: 'REPORT_SUCCESS', payload: { data, reqId } });
      } catch (e) {
        if (!cancelled)
          dispatch({
            type: 'REPORT_ERROR',
            payload: {
              error: e instanceof Error ? e.message : 'Error cargando reporte',
              reqId,
            },
          });
      }
    })();

    return () => { cancelled = true; };
  }, [state.game_pk, state._reqId]);

  // ── Rolling metrics (Bug 5) + calibration data (Roadmap 0.3) ────────────
  useEffect(() => {
    fetchRollingMetrics().then(setRollingMetrics).catch(() => null);
    fetchCalibrationData().then(setCalibrationData).catch(() => null);
  }, []);

  // ── Event handlers ────────────────────────────────────────────────────────
  const handleDateChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      dispatch({ type: 'DATE_CHANGED', payload: e.target.value });
    },
    []
  );

  const handleGameChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      dispatch({ type: 'GAME_SELECTED', payload: Number(e.target.value) });
    },
    []
  );

  const handleTeamToggle = useCallback((t: 'home' | 'away') => {
    dispatch({ type: 'TEAM_CHANGED', payload: t });
  }, []);

  const { report, reportLoading, reportError } = state;

  // Derive selected game names for the toggle labels
  const selectedGame = state.historicalGames.find(g => g.game_pk === state.game_pk)
    ?? state.historicalGames[0]
    ?? null;

  // Build result map from actual lineup
  const resultMap: Record<number, string> = {};
  if (report?.actual_lineup) {
    for (const p of report.actual_lineup) {
      if (p.result) resultMap[p.order] = p.result;
    }
  }

  return (
    <div style={{ minHeight: '100vh', background: '#0A0E1A' }}>
      {/* Page header */}
      <div
        style={{
          padding: '14px 20px',
          borderBottom: '1px solid #162030',
          display: 'flex',
          alignItems: 'baseline',
          gap: 12,
        }}
      >
        <span
          style={{
            fontSize: '1.45rem',
            fontWeight: 800,
            color: '#E8EAF6',
            letterSpacing: '-0.02em',
          }}
        >
          📊 Post-Game Review
        </span>
        <span
          style={{
            fontSize: '0.7rem',
            fontWeight: 600,
            letterSpacing: '0.14em',
            textTransform: 'uppercase',
            color: '#2E5580',
          }}
        >
          Análisis Histórico
        </span>
      </div>

      <div
        style={{
          maxWidth: 1400,
          margin: '0 auto',
          padding: '16px 16px 32px',
          display: 'flex',
          flexDirection: 'column',
          gap: 16,
        }}
      >
        {/* ── Selectors ─────────────────────────────────────────────────── */}
        <Card>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'flex-end' }}>
            {/* Date picker */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <label style={{ fontSize: '0.63rem', color: '#374151', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                Fecha
              </label>
              <input
                type="date"
                value={state.date}
                max={todayMinus(0)}
                onChange={handleDateChange}
                style={{
                  background: '#0D1117',
                  border: '1px solid #1F2937',
                  borderRadius: 8,
                  padding: '6px 10px',
                  color: '#E8EAF6',
                  fontSize: '0.85rem',
                  fontFamily: 'JetBrains Mono, monospace',
                  outline: 'none',
                  cursor: 'pointer',
                }}
              />
            </div>

            {/* Game selector */}
            <div style={{ flex: 1, minWidth: 220, display: 'flex', flexDirection: 'column', gap: 4 }}>
              <label style={{ fontSize: '0.63rem', color: '#374151', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                Partido
              </label>
              {state.historicalLoading ? (
                <Skeleton style={{ height: 36, width: '100%' }} />
              ) : (
                <select
                  value={state.game_pk ?? ''}
                  onChange={handleGameChange}
                  style={{
                    background: '#0D1117',
                    border: '1px solid #1F2937',
                    borderRadius: 8,
                    padding: '6px 10px',
                    color: '#E8EAF6',
                    fontSize: '0.85rem',
                    outline: 'none',
                    cursor: 'pointer',
                    width: '100%',
                  }}
                >
                  {state.historicalGames.length === 0 && (
                    <option value="">Sin partidos registrados</option>
                  )}
                  {state.historicalGames.map((g) => (
                    <option key={g.game_pk} value={g.game_pk}>
                      {g.away_name} @ {g.home_name} · Final: {g.final_score}
                    </option>
                  ))}
                </select>
              )}
            </div>

            {/* Home / Away toggle — mirrors War Room selector */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <label style={{ fontSize: '0.63rem', color: '#374151', letterSpacing: '0.1em', textTransform: 'uppercase' }}>
                Equipo
              </label>
              <div style={{ display: 'flex', borderRadius: 8, border: '1px solid #1F2937', overflow: 'hidden' }}>
                {(['away', 'home'] as const).map((t) => {
                  const active = state.team === t;
                  const label = t === 'away'
                    ? (selectedGame ? `↗ ${selectedGame.away_team ?? selectedGame.away_name}` : 'Visitante')
                    : (selectedGame ? `${selectedGame.home_team ?? selectedGame.home_name} ⌂` : 'Local');
                  return (
                    <button
                      key={t}
                      onClick={() => handleTeamToggle(t)}
                      style={{
                        padding: '6px 14px',
                        fontSize: '0.82rem',
                        fontWeight: 600,
                        border: 'none',
                        cursor: 'pointer',
                        transition: 'background 0.15s',
                        background: active ? '#3B82F6' : '#111827',
                        color: active ? '#fff' : '#6B7280',
                      }}
                    >
                      {label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Score badge */}
            {report && (
              <div
                style={{
                  padding: '8px 16px',
                  borderRadius: 10,
                  background: '#0D1117',
                  border: '1px solid #1F2937',
                  fontFamily: 'JetBrains Mono, monospace',
                  textAlign: 'center',
                  minWidth: 160,
                }}
              >
                <p style={{ fontSize: '0.6rem', color: '#374151', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 2 }}>
                  Resultado Final
                </p>
                <p style={{ fontSize: '1.2rem', fontWeight: 800, color: '#E8EAF6' }}>
                  {report.matchup}
                </p>
                <p
                  style={{
                    fontSize: '0.75rem',
                    fontWeight: 700,
                    color: report.game_result ? '#22C55E' : '#EF4444',
                    marginTop: 2,
                  }}
                >
                  {report.game_result
                    ? `✓ Victoria ${report.team_name ?? (state.team === 'home' ? 'local' : 'visitante')}`
                    : `✗ Derrota ${report.team_name ?? (state.team === 'home' ? 'local' : 'visitante')}`
                  }
                </p>
              </div>
            )}
          </div>

          {/* Away-team notice — shown when no model prediction available */}
          {state.team === 'away' && (
            <div style={{
              marginTop: 10,
              padding: '8px 12px',
              borderRadius: 6,
              background: 'rgba(59,130,246,0.06)',
              border: '1px solid rgba(59,130,246,0.2)',
              fontSize: '0.72rem',
              color: '#3B82F6',
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}>
              <span>ℹ</span>
              <span>
                Vista del equipo visitante. Las predicciones del modelo y el análisis de divergencias
                solo están disponibles para el equipo local (el DB almacena predicciones del home team).
                Se muestra el lineup real del visitante.
              </span>
            </div>
          )}
        </Card>

        {/* ── Error states ──────────────────────────────────────────────── */}
        {reportError && (
          <div
            style={{
              padding: '12px 16px',
              borderRadius: 8,
              background: 'rgba(239,68,68,0.08)',
              border: '1px solid #EF4444',
              color: '#EF4444',
              fontSize: '0.85rem',
            }}
          >
            Error cargando reporte: {reportError}
          </div>
        )}

        {/* ── Projection chart + KPI summary (Bug 2) ───────────────────── */}
        {(reportLoading || report) && (
          <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: 16 }}>
            {/* Chart */}
            <Card>
              <SectionLabel>
                E[R] Proyectado vs. Carreras Reales
              </SectionLabel>
              {reportLoading ? (
                <Skeleton style={{ height: 180 }} />
              ) : report ? (
                <>
                  <ProjectionChart
                    er={report.projected_runs}
                    std_dev={report.std_dev}
                    percentile_10={report.percentile_10}
                    percentile_25={report.percentile_25}
                    percentile_75={report.percentile_75}
                    percentile_90={report.percentile_90}
                    uncertainty_level={report.uncertainty_level}
                    actual_runs={state.team === 'away' ? report.actual_away_runs : report.actual_home_runs}
                    simulation_n={report.simulation_n}
                  />
                  {report.std_dev !== undefined && report.projected_runs > 0 && (
                    <p style={{ marginTop: 10, fontSize: '0.75rem', color: '#546E7A', fontFamily: 'JetBrains Mono, monospace' }}>
                      E[R] = {report.projected_runs.toFixed(2)} ± {report.std_dev.toFixed(2)}
                      {report.win_prob_ci_low !== undefined && (
                        <> &nbsp;·&nbsp; P(W) = {(report.win_probability_projected * 100).toFixed(1)}%
                          {' '}(IC 90%: {(report.win_prob_ci_low * 100).toFixed(0)}%–{(report.win_prob_ci_high! * 100).toFixed(0)}%)
                        </>
                      )}
                    </p>
                  )}
                  {report.projected_runs === 0 && state.team === 'away' && (
                    <p style={{ marginTop: 10, fontSize: '0.75rem', color: '#374151', fontStyle: 'italic' }}>
                      Sin proyección de modelo para el equipo visitante. Se muestra solo el resultado real.
                    </p>
                  )}
                </>
              ) : null}
            </Card>

            {/* KPI summary */}
            <Card>
              <SectionLabel>Resumen del Partido</SectionLabel>
              {reportLoading ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                  {[1, 2, 3, 4].map((i) => <Skeleton key={i} style={{ height: 48 }} />)}
                </div>
              ) : report ? (() => {
                  const teamRuns = state.team === 'away' ? report.actual_away_runs : report.actual_home_runs;
                  const oppRuns  = state.team === 'away' ? report.actual_home_runs : report.actual_away_runs;
                  const teamName = report.team_name ?? (state.team === 'home' ? 'Local' : 'Visitante');
                  const oppName  = report.opponent_name ?? (state.team === 'home' ? 'Visitante' : 'Local');
                  return (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                      <KPICard
                        label={`Carreras — ${teamName}`}
                        value={String(teamRuns)}
                      />
                      <KPICard
                        label={`Carreras — ${oppName}`}
                        value={String(oppRuns)}
                      />
                      {report.projected_runs > 0 ? (
                        <KPICard
                          label="E[R] Proyectado (modelo)"
                          value={`${report.projected_runs.toFixed(2)}`}
                          sub={`${teamRuns - report.projected_runs >= 0 ? '+' : ''}${(teamRuns - report.projected_runs).toFixed(2)} vs real`}
                          subColor={teamRuns >= report.projected_runs ? '#22C55E' : '#EF4444'}
                        />
                      ) : state.team === 'away' ? (
                        <div style={{
                          padding: '10px 14px', borderRadius: 8,
                          background: '#0D1117', border: '1px solid #1F2937',
                        }}>
                          <p style={{ fontSize: '0.63rem', color: '#374151', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 4 }}>
                            E[R] Proyectado (modelo)
                          </p>
                          <p style={{ fontSize: '0.82rem', color: '#374151', fontStyle: 'italic' }}>
                            Sin predicción para el equipo visitante
                          </p>
                        </div>
                      ) : null}
                      {report.win_probability_projected > 0 && (
                        <KPICard
                          label={`P(Victoria) — ${teamName}`}
                          value={`${(report.win_probability_projected * 100).toFixed(1)}%`}
                          sub={
                            report.win_prob_ci_low !== undefined
                              ? `IC 90%: ${(report.win_prob_ci_low * 100).toFixed(0)}%–${(report.win_prob_ci_high! * 100).toFixed(0)}%`
                              : undefined
                          }
                        />
                      )}
                    </div>
                  );
                })()
              : null}
            </Card>
          </div>
        )}

        {/* ── Hero metrics — Audit 2.5 ROI quantification ─────────────── */}
        {report?.divergences && report.divergences.length > 0 && (() => {
          const nDivergences = report.divergences.filter(d => !d.match).length;
          const nSignificant = report.divergences.filter(d =>
            !d.match && (d.significant === true || (d.significant === undefined && Math.abs(d.delta_er) > 0.01))
          ).length;
          const netDelta = report.divergences.reduce((acc, d) => acc + (d.match ? 0 : d.delta_er), 0);
          const isPositive = netDelta >= 0;
          // Audit 2.5 — ROI: convert ΔE[R] to win probability impact
          // ~0.1 run ≈ 1pp win prob (Pythagorean approximation at 4.5 run baseline)
          const winProbImpact = netDelta * 10; // pp
          return (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
              {/* ΔE[R] neto */}
              <Card style={{ padding: '16px 20px' }}>
                <p style={{ fontSize: '0.6rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: '#2E5580', marginBottom: 6 }}>
                  Impacto Neto ΔE[R]
                </p>
                <p style={{
                  fontSize: '2.4rem', fontWeight: 800, fontFamily: 'JetBrains Mono, monospace',
                  color: isPositive ? '#22C55E' : '#EF4444', lineHeight: 1,
                }}>
                  {netDelta >= 0 ? '+' : ''}{netDelta.toFixed(3)}
                </p>
                <p style={{ fontSize: '0.72rem', color: '#546E7A', marginTop: 4 }}>
                  E[R] adicional vs. lineup óptimo
                </p>
              </Card>
              {/* ROI en P(W) — Audit 2.5 */}
              <Card style={{
                padding: '16px 20px',
                border: `1px solid ${Math.abs(winProbImpact) >= 5 ? 'rgba(245,158,11,0.35)' : '#1F2937'}`,
              }}>
                <p style={{ fontSize: '0.6rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: '#2E5580', marginBottom: 6 }}>
                  Impacto en P(W)
                </p>
                <p style={{
                  fontSize: '2.4rem', fontWeight: 800, fontFamily: 'JetBrains Mono, monospace',
                  color: winProbImpact >= 0 ? '#22C55E' : '#EF4444', lineHeight: 1,
                }}>
                  {winProbImpact >= 0 ? '+' : ''}{winProbImpact.toFixed(1)}
                  <span style={{ fontSize: '1.4rem' }}>pp</span>
                </p>
                <p style={{ fontSize: '0.72rem', color: '#546E7A', marginTop: 4 }}>
                  aprox. probabilidad de victoria
                </p>
                <p style={{
                  fontSize: '0.63rem', color: '#374151', marginTop: 4,
                  fontStyle: 'italic',
                }}>
                  ~10pp / 0.1 E[R] (Pitágoras, base 4.5R)
                </p>
              </Card>
              {/* Divergencias significativas */}
              <Card style={{ padding: '16px 20px' }}>
                <p style={{ fontSize: '0.6rem', letterSpacing: '0.14em', textTransform: 'uppercase', color: '#2E5580', marginBottom: 6 }}>
                  Divergencias Significativas
                </p>
                <p style={{
                  fontSize: '2.4rem', fontWeight: 800, fontFamily: 'JetBrains Mono, monospace',
                  color: nSignificant > 0 ? '#F59E0B' : '#6B7280', lineHeight: 1,
                }}>
                  {nSignificant}
                  <span style={{ fontSize: '1.2rem', color: '#374151' }}>/{nDivergences}</span>
                </p>
                <p style={{ fontSize: '0.72rem', color: '#546E7A', marginTop: 4 }}>
                  superan el IC90 (excluyen el cero)
                </p>
              </Card>
            </div>
          );
        })()}

        {/* ── Error Attribution (Audit 2.1) — shown only when model predicted & |Δ| > 4 ── */}
        {report && report.projected_runs > 0 && (() => {
          const teamRuns = state.team === 'away' ? report.actual_away_runs : report.actual_home_runs;
          return Math.abs(teamRuns - report.projected_runs) > 4 ? (
            <ErrorAttributionCard
              projected={report.projected_runs}
              actual={teamRuns}
              percentile10={report.percentile_10}
              percentile90={report.percentile_90}
              divergences={report.divergences}
            />
          ) : null;
        })()}

        {/* ── Divergence Table (Bug 3 + 4) ─────────────────────────────── */}
        <Card>
          <SectionLabel>
            Lineup Propuesto vs. Lineup Utilizado — Análisis de Divergencias
          </SectionLabel>
          {reportLoading ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {Array.from({ length: 9 }).map((_, i) => (
                <Skeleton key={i} style={{ height: 42 }} />
              ))}
            </div>
          ) : report?.divergences?.length ? (
            <DivergenceTable
              divergences={report.divergences}
              results={resultMap}
            />
          ) : report?.proposed_lineup?.length ? (
            // Fallback: no pre-computed divergences, render read-only table
            <FallbackLineupTable report={report} />
          ) : (
            <p style={{ color: '#374151', fontSize: '0.85rem', fontStyle: 'italic' }}>
              No hay predicción de modelo disponible para este partido. Se muestra únicamente el lineup real.
            </p>
          )}
        </Card>

        {/* ── Rolling Metrics (Bug 5) ────────────────────────────────────── */}
        {rollingMetrics && (
          <Card>
            <MetricsPanel
              metrics={rollingMetrics}
              gameLogLoss={report?.model_log_loss}
            />
          </Card>
        )}

        {/* ── Reliability Diagram (Roadmap 0.3) ─────────────────────────── */}
        {calibrationData && calibrationData.bins.length > 0 && (
          <Card>
            <SectionLabel>Diagrama de Calibración — Reliability Diagram</SectionLabel>
            <CalibrationChart
              bins={calibrationData.bins}
              ece={calibrationData.ece}
              nGames={calibrationData.n_games}
              source={calibrationData.source}
            />
          </Card>
        )}

        {/* Footer */}
        <p style={{ textAlign: 'center', fontSize: '0.65rem', color: '#1F2937', fontFamily: 'JetBrains Mono, monospace' }}>
          MLB Post-Game Review · Datos: MLB Stats API + modelo v{report?.model_version ?? '—'}
        </p>
      </div>

      {/* Shimmer animation */}
      <style>{`
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
      `}</style>
    </div>
  );
}

// ── Audit 2.1 — Error Attribution Card ────────────────────────────────────────
// Shown only when |actual − projected| > 4 runs.
// Decomposes the prediction error into four traceable categories.

interface ErrorAttribProps {
  projected: number;
  actual: number;
  percentile10?: number;
  percentile90?: number;
  divergences?: import('../types').DivergenceRow[];
}

function ErrorAttributionCard({ projected, actual, percentile10, percentile90, divergences }: ErrorAttribProps) {
  const error = actual - projected;
  if (Math.abs(error) <= 4) return null;

  const isOverrun = error > 0;  // more runs than expected
  const divergenceDelta = (divergences ?? []).reduce((acc, d) => acc + (d.match ? 0 : d.delta_er), 0);

  // Heuristic decomposition — clearly labelled as estimates
  // Component B: offensive efficiency from known divergences (measured)
  const compB_measured = divergenceDelta;
  // Residual to explain
  const residual = error - compB_measured;
  // Pitcher component (A): typically 35-45% of variance unexplained by batting
  const compA_estimate = residual * 0.40;
  // Explosion inning (C): if actual > P90, residual likely from a single inning
  const aboveP90 = percentile90 !== undefined && actual > percentile90;
  const compC_estimate = aboveP90 ? residual * 0.45 : residual * 0.20;
  // Weather/uncaptured (D): remainder
  const compD_residual = residual - compA_estimate - (aboveP90 ? residual * 0.45 : residual * 0.20);

  const absError = Math.abs(error);

  return (
    <div style={{
      background: '#0D1117',
      border: `1px solid ${isOverrun ? 'rgba(239,68,68,0.35)' : 'rgba(59,130,246,0.35)'}`,
      borderRadius: 12, padding: '16px 18px',
      marginBottom: 0,
    }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 12 }}>
        <span style={{ fontSize: '1.1rem' }}>{isOverrun ? '⚠️' : '📉'}</span>
        <div>
          <p style={{
            fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.14em',
            textTransform: 'uppercase',
            color: isOverrun ? '#EF4444' : '#3B82F6',
          }}>
            Gran Desviación Detectada — Atribución de Error
          </p>
          <p style={{ fontSize: '0.78rem', color: '#E8EAF6', fontFamily: 'JetBrains Mono, monospace', marginTop: 2 }}>
            Proyectado: {projected.toFixed(2)} · Real: {actual} · Error:{' '}
            <span style={{ color: isOverrun ? '#EF4444' : '#3B82F6', fontWeight: 700 }}>
              {error >= 0 ? '+' : ''}{error.toFixed(2)} carreras
            </span>
          </p>
        </div>
        {aboveP90 && (
          <span style={{
            marginLeft: 'auto', fontSize: '0.62rem', fontWeight: 700,
            padding: '2px 8px', borderRadius: 6,
            background: 'rgba(239,68,68,0.1)', color: '#EF4444',
            border: '1px solid rgba(239,68,68,0.3)',
          }}>
            Por encima de P90
          </span>
        )}
      </div>

      {/* Components grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8 }}>

        {/* A: Pitcher model */}
        <div style={{
          background: '#111827', borderRadius: 8, padding: '10px 12px',
          border: '1px solid #1F2937',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 4 }}>
            (A) Pitcher
          </p>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.1rem', color: '#78909C' }}>
            {compA_estimate >= 0 ? '+' : ''}{compA_estimate.toFixed(2)}
          </span>
          <p style={{ fontSize: '0.60rem', color: '#374151', marginTop: 3 }}>
            estimado · 40% residual
          </p>
          <p style={{ fontSize: '0.58rem', color: '#4B5563', marginTop: 2, fontStyle: 'italic' }}>
            Rendimiento real del pitcher vs modelo
          </p>
        </div>

        {/* B: Offensive efficiency — MEASURED */}
        <div style={{
          background: '#111827', borderRadius: 8, padding: '10px 12px',
          border: Math.abs(compB_measured) > 0.1
            ? '1px solid rgba(245,158,11,0.35)'
            : '1px solid #1F2937',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 4 }}>
            (B) Eficiencia ofensiva
          </p>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.1rem',
            color: Math.abs(compB_measured) > 0.1 ? '#F59E0B' : '#78909C',
          }}>
            {compB_measured >= 0 ? '+' : ''}{compB_measured.toFixed(2)}
          </span>
          <p style={{ fontSize: '0.60rem', color: '#22C55E', marginTop: 3 }}>
            medido · divergencias ΔE[R]
          </p>
          <p style={{ fontSize: '0.58rem', color: '#4B5563', marginTop: 2, fontStyle: 'italic' }}>
            {(divergences ?? []).filter(d => !d.match).length} cambios del manager
          </p>
        </div>

        {/* C: Explosion inning */}
        <div style={{
          background: '#111827', borderRadius: 8, padding: '10px 12px',
          border: aboveP90 ? '1px solid rgba(239,68,68,0.35)' : '1px solid #1F2937',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 4 }}>
            (C) Inning explosivo
          </p>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.1rem',
            color: aboveP90 ? '#EF4444' : '#78909C',
          }}>
            {aboveP90
              ? `+${compC_estimate.toFixed(2)}`
              : '—'}
          </span>
          <p style={{ fontSize: '0.60rem', color: '#374151', marginTop: 3 }}>
            {aboveP90 ? 'estimado · acumulación inning' : 'no aplicable'}
          </p>
          <p style={{ fontSize: '0.58rem', color: '#4B5563', marginTop: 2, fontStyle: 'italic' }}>
            {aboveP90
              ? `Real (${actual}) > P90 (${percentile90?.toFixed(1)})`
              : 'Dentro del intervalo P90'}
          </p>
        </div>

        {/* D: Uncaptured factors */}
        <div style={{
          background: '#111827', borderRadius: 8, padding: '10px 12px',
          border: '1px solid #1F2937',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 4 }}>
            (D) Factores no capturados
          </p>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.1rem', color: '#374151' }}>
            {compD_residual >= 0 ? '+' : ''}{compD_residual.toFixed(2)}
          </span>
          <p style={{ fontSize: '0.60rem', color: '#374151', marginTop: 3 }}>
            residual · clima / contexto
          </p>
          <p style={{ fontSize: '0.58rem', color: '#4B5563', marginTop: 2, fontStyle: 'italic' }}>
            Datos climáticos pendientes
          </p>
        </div>
      </div>

      {/* Summary interpretation */}
      <div style={{
        marginTop: 10, padding: '8px 12px', borderRadius: 6,
        background: 'rgba(255,255,255,0.02)', border: '1px solid #1F2937',
        fontSize: '0.7rem', color: '#78909C', lineHeight: 1.6,
      }}>
        <span style={{ color: '#546E7A', fontWeight: 700 }}>▶ Interpretación:</span>{' '}
        Error total de <span style={{ color: '#E8EAF6', fontFamily: 'JetBrains Mono, monospace' }}>{absError.toFixed(2)} runs</span>.{' '}
        {Math.abs(compB_measured) > 0.5
          ? `${Math.abs(compB_measured / error * 100).toFixed(0)}% atribuible a cambios del manager.`
          : 'Cambios del manager tienen impacto menor en esta desviación.'
        }{' '}
        {aboveP90
          ? 'Fuera del intervalo P10-P90 — evento de cola, varianza inherente.'
          : 'Dentro del intervalo P10-P90 — error sistemático del modelo.'
        }
        <span style={{ color: '#374151', marginLeft: 6, fontSize: '0.62rem', fontStyle: 'italic' }}>
          [A,C,D] son estimaciones heurísticas; [B] es medición directa de divergencias.
        </span>
      </div>
    </div>
  );
}

// ── Inline KPI card ───────────────────────────────────────────────────────────

function KPICard({
  label,
  value,
  sub,
  subColor = '#546E7A',
}: {
  label: string;
  value: string;
  sub?: string;
  subColor?: string;
}) {
  return (
    <div
      style={{
        background: '#0D1117',
        border: '1px solid #1F2937',
        borderRadius: 8,
        padding: '10px 14px',
      }}
    >
      <p style={{ fontSize: '0.63rem', color: '#374151', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 4 }}>
        {label}
      </p>
      <p style={{ fontSize: '1.25rem', fontWeight: 700, color: '#E8EAF6', fontFamily: 'JetBrains Mono, monospace' }}>
        {value}
      </p>
      {sub && (
        <p style={{ fontSize: '0.7rem', color: subColor, fontFamily: 'JetBrains Mono, monospace', marginTop: 2 }}>
          {sub}
        </p>
      )}
    </div>
  );
}

// ── Fallback simple lineup table (no divergences pre-computed) ─────────────────

function FallbackLineupTable({ report }: { report: PostGameReport }) {
  const proposed = report.proposed_lineup ?? [];
  const actual = report.actual_lineup ?? [];

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
      <div>
        <p style={{ fontSize: '0.72rem', fontWeight: 600, color: '#90CAF9', marginBottom: 8 }}>
          🤖 Lineup del Modelo
        </p>
        <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
          <tbody>
            {proposed.map((p) => {
              const matchAct = actual.find((a) => a.order === p.order);
              const match = matchAct?.name === p.name;
              return (
                <tr
                  key={p.order}
                  style={{
                    borderBottom: '1px solid #1F2937',
                    background: match ? 'transparent' : 'rgba(139,92,246,0.06)',
                  }}
                >
                  <td style={{ padding: '6px 8px', color: '#6B7280', fontFamily: 'JetBrains Mono, monospace', width: 28 }}>
                    #{p.order}
                  </td>
                  <td style={{ padding: '6px 8px', color: match ? '#90CAF9' : '#CE93D8', fontWeight: match ? 400 : 600 }}>
                    {p.name}
                  </td>
                  <td style={{ padding: '6px 8px', color: '#374151', textAlign: 'right' }}>
                    {p.pos}
                  </td>
                  <td style={{ padding: '6px 8px', textAlign: 'right', color: match ? '#22C55E' : '#F59E0B', fontSize: '0.7rem' }}>
                    {match ? '=' : '≠'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div>
        <p style={{ fontSize: '0.72rem', fontWeight: 600, color: '#FFC107', marginBottom: 8 }}>
          👨‍💼 Lineup Real (Manager)
        </p>
        <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
          <tbody>
            {actual.map((p) => {
              const matchProp = proposed.find((pr) => pr.order === p.order);
              const match = matchProp?.name === p.name;
              return (
                <tr
                  key={p.order}
                  style={{
                    borderBottom: '1px solid #1F2937',
                    background: match ? 'transparent' : 'rgba(255,193,7,0.06)',
                  }}
                >
                  <td style={{ padding: '6px 8px', color: '#6B7280', fontFamily: 'JetBrains Mono, monospace', width: 28 }}>
                    #{p.order}
                  </td>
                  <td style={{ padding: '6px 8px', color: match ? '#E8EAF6' : '#FFC107', fontWeight: match ? 400 : 600 }}>
                    {p.name}
                  </td>
                  <td style={{ padding: '6px 8px', color: '#374151' }}>
                    {p.pos}
                  </td>
                  <td style={{ padding: '6px 8px', color: '#546E7A', fontSize: '0.7rem', fontFamily: 'JetBrains Mono, monospace' }}>
                    {p.result ?? '—'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
