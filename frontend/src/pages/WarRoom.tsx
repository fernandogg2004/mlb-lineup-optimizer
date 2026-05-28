import { useState, useCallback, useEffect } from 'react';
import { clsx } from 'clsx';
import type { Player, WhatIfResult, Precision30d } from '../types';
import { useGames } from '../hooks/useGames';
import { useGameContext } from '../context/GameContext';
import Header from '../components/layout/Header';
import WeatherBar from '../components/ui/WeatherBar';
import RunsDistribution from '../components/kpi/RunsDistribution';
import BulletKPI from '../components/kpi/BulletKPI';
import UncertaintyPanel from '../components/kpi/UncertaintyPanel';
import WinProbXAI from '../components/kpi/WinProbXAI';
import OPSChart from '../components/charts/OPSChart';
import ZScoreChart from '../components/charts/ZScoreChart';
import LineupTable from '../components/lineup/LineupTable';
import WhatIfPanel from '../components/lineup/WhatIfPanel';
import IntelBriefing from '../components/intel/IntelBriefing';
import MatchupContextPanel from '../components/kpi/MatchupContextPanel';

// Skeleton loader component
function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={clsx('animate-pulse rounded-lg', className)}
      style={{ background: '#1a2035' }}
    />
  );
}

function Card({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={clsx('rounded-xl border p-5', className)}
      style={{ background: '#111827', borderColor: '#1F2937' }}
    >
      {children}
    </div>
  );
}

// ── Audit 1.8 — Precision Breakdown Band ────────────────────────────────────
// Shows Accuracy / MAE / Brier Score breakdown with ELO benchmark comparison.
// Only renders when the API returns `precision_30d` data.
function PrecisionBand({ precision }: { precision: Precision30d }) {
  const brierBetter = precision.brier_score <= precision.brier_benchmark;
  const eloBetter   = precision.vs_elo_delta >= 0;
  const maePoor     = precision.mae_carreras > 2.0;

  return (
    <div
      style={{
        background: '#0D1117',
        border: '1px solid #1F2937',
        borderRadius: 10,
        padding: '10px 16px',
        display: 'grid',
        gridTemplateColumns: 'repeat(4, 1fr)',
        gap: 8,
      }}
    >
      {/* Accuracy */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <p style={{
          fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: '#374151',
        }}>
          Accuracy 30d
        </p>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace', fontWeight: 800, fontSize: '1.25rem',
          color: precision.accuracy_pct >= 60 ? '#22C55E' : precision.accuracy_pct >= 50 ? '#F59E0B' : '#EF4444',
        }}>
          {precision.accuracy_pct.toFixed(1)}%
        </span>
        <p style={{ fontSize: '0.63rem', color: '#374151' }}>
          ganador correcto
          {eloBetter && (
            <span style={{ marginLeft: 4, color: '#22C55E', fontSize: '0.60rem' }}>
              +{precision.vs_elo_delta.toFixed(1)}pp vs ELO
            </span>
          )}
        </p>
      </div>

      {/* MAE */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <p style={{
          fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: '#374151',
        }}>
          MAE E[R]
        </p>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace', fontWeight: 800, fontSize: '1.25rem',
          color: maePoor ? '#F59E0B' : '#22C55E',
        }}>
          {precision.mae_carreras.toFixed(2)}
        </span>
        <p style={{ fontSize: '0.63rem', color: '#374151' }}>
          carreras · error medio absoluto
        </p>
      </div>

      {/* Brier Score */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <p style={{
          fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: '#374151',
        }}>
          Brier Score
        </p>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 6 }}>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontWeight: 800, fontSize: '1.25rem',
            color: brierBetter ? '#22C55E' : '#F59E0B',
          }}>
            {precision.brier_score.toFixed(3)}
          </span>
          <span style={{
            fontSize: '0.60rem', fontWeight: 700, padding: '1px 5px', borderRadius: 4,
            background: brierBetter ? 'rgba(34,197,94,0.1)' : 'rgba(245,158,11,0.1)',
            color: brierBetter ? '#22C55E' : '#F59E0B',
            border: `1px solid ${brierBetter ? 'rgba(34,197,94,0.25)' : 'rgba(245,158,11,0.25)'}`,
          }}>
            {brierBetter ? '▼ mejor' : '▲ alto'}
          </span>
        </div>
        <p style={{ fontSize: '0.63rem', color: '#374151' }}>
          benchmark ELO: {precision.brier_benchmark.toFixed(3)}
        </p>
      </div>

      {/* n partidos */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        <p style={{
          fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.12em',
          textTransform: 'uppercase', color: '#374151',
        }}>
          Muestra
        </p>
        <span style={{
          fontFamily: 'JetBrains Mono, monospace', fontWeight: 800, fontSize: '1.25rem',
          color: precision.n_games >= 30 ? '#E8EAF6' : '#F59E0B',
        }}>
          {precision.n_games}
        </span>
        <p style={{ fontSize: '0.63rem', color: '#374151' }}>
          partidos evaluados (30d)
          {precision.n_games < 30 && (
            <span style={{ color: '#F59E0B', marginLeft: 4 }}>⚠ n&lt;30</span>
          )}
        </p>
      </div>
    </div>
  );
}

export default function WarRoom() {
  const { games, loading: gamesLoading, error: gamesError } = useGames();
  // Bug 1 fix: all lineup state lives in GameContext with strict requestId invalidation.
  const { state: gameState, dispatch } = useGameContext();
  const [whatIfResult, setWhatIfResult] = useState<WhatIfResult | null>(null);
  const [localLineup, setLocalLineup] = useState<Player[] | null>(null);

  const lineupData   = gameState.lineupData;
  const lineupLoading = gameState.lineupLoading;
  const lineupError  = gameState.lineupError;
  const effectiveGamePk = gameState.game_pk ?? (games.length > 0 ? games[0].game_pk : null);
  const team = gameState.team;

  // Auto-select first game when list loads
  useEffect(() => {
    if (games.length > 0 && gameState.game_pk === null) {
      dispatch({ type: 'GAME_CHANGED', payload: { game_pk: games[0].game_pk } });
    }
  }, [games, gameState.game_pk, dispatch]);

  const selectedGame =
    games.find((g) => g.game_pk === effectiveGamePk) ?? games[0] ?? null;

  // Sync local lineup when API data arrives for a new game
  useEffect(() => {
    setLocalLineup(null);
  }, [effectiveGamePk, team]);

  const currentLineup = localLineup ?? lineupData?.lineup ?? [];

  const handleGameChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      // Dispatch triggers reducer: lineup = null, loading = true (Bug 1)
      dispatch({ type: 'GAME_CHANGED', payload: { game_pk: Number(e.target.value) } });
      setLocalLineup(null);
      setWhatIfResult(null);
    },
    [dispatch]
  );

  const handleTeamToggle = useCallback((t: 'home' | 'away') => {
    dispatch({ type: 'TEAM_CHANGED', payload: { team: t } });
    setLocalLineup(null);
    setWhatIfResult(null);
  }, [dispatch]);

  const handleReorder = useCallback((newLineup: Player[]) => {
    setLocalLineup(newLineup);
  }, []);

  const handleWhatIfResult = useCallback((result: WhatIfResult) => {
    setWhatIfResult(result);
  }, []);

  const handleReset = useCallback(() => {
    setLocalLineup(lineupData?.lineup ?? null);
    setWhatIfResult(null);
  }, [lineupData]);

  const pitcher = selectedGame
    ? team === 'home'
      ? { name: selectedGame.away_pitcher, hand: selectedGame.away_pitcher_hand }
      : { name: selectedGame.home_pitcher, hand: selectedGame.home_pitcher_hand }
    : null;

  const er = lineupData?.expected_runs ?? 4.5;
  const winProb = (lineupData?.win_probability ?? 0.5) * 100;
  // away_win_probability es el complemento exacto (1 − win_probability) calculado
  // desde la misma simulación; si el campo no viene del API se deriva aquí para
  // garantizar que ambos valores siempre sumen 100 %.
  const awayWinProb = lineupData
    ? ((lineupData.away_win_probability ?? (1 - (lineupData.win_probability ?? 0.5))) * 100)
    : 50;
  const confidence = (lineupData?.model_confidence ?? 0.7) * 100;

  return (
    <div className="min-h-screen" style={{ background: '#0A0E1A' }}>
      <Header />

      {/* Weather Bar */}
      <WeatherBar
        venue={selectedGame?.venue ?? '—'}
        pitcher={pitcher?.name ?? '—'}
        pitcherHand={pitcher?.hand ?? 'R'}
        pitcherERA={lineupData?.pitcher_era ?? 0}
      />

      <div className="px-4 py-4 max-w-[1600px] mx-auto space-y-4">
        {/* Errors */}
        {gamesError && (
          <div
            className="rounded-lg border px-4 py-3 text-sm"
            style={{
              background: 'rgba(255,69,96,0.08)',
              borderColor: '#FF4560',
              color: '#FF4560',
            }}
          >
            Error cargando juegos: {gamesError}
          </div>
        )}

        {lineupError && (
          <div
            className="rounded-lg border px-4 py-3 text-sm"
            style={{
              background: 'rgba(255,69,96,0.08)',
              borderColor: '#FF4560',
              color: '#FF4560',
            }}
          >
            Error cargando datos del partido: {lineupError}
          </div>
        )}

        {/* Game Selector Row */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Game Dropdown */}
          <div className="flex-1 min-w-[200px] max-w-sm">
            {gamesLoading ? (
              <Skeleton className="h-10 w-full" />
            ) : (
              <select
                value={effectiveGamePk ?? ''}
                onChange={handleGameChange}
                className="w-full h-10 rounded-lg px-3 text-sm font-mono text-white border focus:outline-none focus:ring-1"
                style={{
                  background: '#111827',
                  borderColor: '#1F2937',
                  color: '#E8EAF6',
                }}
                onFocus={(e: React.FocusEvent<HTMLSelectElement>) => {
                  e.currentTarget.style.borderColor = '#3B82F6';
                }}
                onBlur={(e: React.FocusEvent<HTMLSelectElement>) => {
                  e.currentTarget.style.borderColor = '#1F2937';
                }}
              >
                {games.length === 0 && (
                  <option value="">Sin juegos disponibles</option>
                )}
                {games.map((g) => (
                  <option key={g.game_pk} value={g.game_pk}>
                    {g.away_team} @ {g.home_team} · {g.game_time}
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Home/Away Toggle */}
          <div
            className="flex rounded-lg border overflow-hidden"
            style={{ borderColor: '#1F2937' }}
          >
            {(['away', 'home'] as const).map((t) => (
              <button
                key={t}
                onClick={() => handleTeamToggle(t)}
                className="px-4 py-2 text-sm font-semibold transition-colors"
                style={
                  team === t
                    ? { background: '#3B82F6', color: '#fff' }
                    : {
                        background: '#111827',
                        color: '#6B7280',
                      }
                }
              >
                {t === 'away'
                  ? selectedGame
                    ? `↗ ${selectedGame.away_team}`
                    : 'Visitante'
                  : selectedGame
                  ? `${selectedGame.home_team} ⌂`
                  : 'Local'}
              </button>
            ))}
          </div>

          {/* Game info badge */}
          {selectedGame && (
            <span className="text-xs font-mono text-slate-500">
              {selectedGame.venue}
            </span>
          )}
        </div>

        {/* KPI Row */}
        <div className="grid grid-cols-3 gap-4">
          {/* Runs Distribution */}
          <Card>
            {/* E[R] as P10–P50–P90 range (Roadmap 1.6 — no false point precision) */}
            {lineupData?.percentile_10 !== undefined && lineupData?.percentile_90 !== undefined ? (
              <div className="flex items-baseline gap-2 mb-3 flex-wrap">
                <span className="text-xl font-mono text-slate-500">{lineupData.percentile_10.toFixed(1)}</span>
                <span className="text-xs text-slate-600">P10</span>
                <span className="text-4xl font-mono font-bold" style={{ color: '#00FF87' }}>
                  {er.toFixed(1)}
                </span>
                <span className="text-xs text-slate-600">P50</span>
                <span className="text-xl font-mono text-slate-500">{lineupData.percentile_90.toFixed(1)}</span>
                <span className="text-xs text-slate-600">P90</span>
                <span className="text-xs text-slate-500 ml-1">E[R] intervalo</span>
              </div>
            ) : (
              <div className="flex items-baseline gap-3 mb-3">
                <span className="text-4xl font-mono font-bold" style={{ color: '#00FF87' }}>
                  {er.toFixed(2)}
                </span>
                <span className="text-sm text-slate-400">E[R] proyectado</span>
              </div>
            )}
            {lineupLoading ? (
              <Skeleton className="h-40 w-full" />
            ) : (
              <RunsDistribution
                er={er}
                percentiles={lineupData?.runs_scored_percentiles ?? {}}
                p10={lineupData?.percentile_10}
                p50={lineupData?.runs_scored_percentiles?.['50']}
                p90={lineupData?.percentile_90}
              />
            )}
          </Card>

          {/* Win Probability — shown as interval when CI available (Roadmap 2.3) */}
          <Card className="flex flex-col justify-center gap-4">
            <p className="text-xs uppercase tracking-widest text-slate-500">
              Probabilidad de Victoria
            </p>
            {lineupLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : (
              <>
                <BulletKPI
                  label="P(Victoria)"
                  value={winProb}
                  unit="%"
                  targetValue={50}
                  targetLabel="50% equilibrio"
                  referenceValue={55}
                  referenceLabel="Umbral sugerido"
                  referenceTooltip="55% es el umbral empírico a partir del cual el modelo históricamente muestra ventaja estadística sobre la línea de mercado (backtesting 30d)."
                />
                {/* Complemento del rival — siempre suma 100% con P(Victoria) */}
                {lineupData && (
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    alignItems: 'center',
                    background: '#0D1117', border: '1px solid #1F2937',
                    borderRadius: 6, padding: '5px 10px',
                    fontSize: '0.72rem', fontFamily: 'JetBrains Mono, monospace',
                  }}>
                    <span style={{ color: '#78909C' }}>
                      P({team === 'home' ? 'Visitante' : 'Local'} gana)
                    </span>
                    <span style={{ color: '#F87171', fontWeight: 600 }}>
                      {awayWinProb.toFixed(1)}%
                    </span>
                    <span style={{ color: '#374151', fontSize: '0.60rem' }}>
                      ∑ = 100 %
                    </span>
                  </div>
                )}
                {/* IC90 interval — not just a point estimate (Roadmap 2.3) */}
                {lineupData?.win_prob_ci_low !== undefined && lineupData?.win_prob_ci_high !== undefined && (
                  <div style={{
                    background: '#0D1117', border: '1px solid #1F2937',
                    borderRadius: 6, padding: '6px 10px',
                    fontSize: '0.72rem', fontFamily: 'JetBrains Mono, monospace',
                    color: '#78909C',
                  }}>
                    IC90:{' '}
                    <span style={{ color: '#90CAF9' }}>
                      {(lineupData.win_prob_ci_low * 100).toFixed(0)}%
                      {' – '}
                      {(lineupData.win_prob_ci_high * 100).toFixed(0)}%
                    </span>
                    <span style={{ color: '#374151', marginLeft: 8, fontSize: '0.62rem' }}>
                      (no solo el punto estimado)
                    </span>
                  </div>
                )}
              </>
            )}
            {lineupData?.win_prob_attribution && lineupData.win_prob_attribution.length > 0 && (
              <div className="mt-2 pt-2 border-t border-slate-800">
                <WinProbXAI
                  factors={lineupData.win_prob_attribution}
                  basePct={lineupData.win_prob_base_pct ?? 50}
                  finalPct={winProb}
                />
              </div>
            )}
            {lineupData && (
              <div className="mt-2 pt-2 border-t border-slate-800">
                <div className="grid grid-cols-2 gap-2 text-xs font-mono text-slate-400">
                  <div>
                    <span className="text-slate-600 uppercase text-[9px]">Modo</span>
                    <p className="text-white">{lineupData.optimization_mode}</p>
                  </div>
                  <div>
                    <span className="text-slate-600 uppercase text-[9px]">Modelo</span>
                    <p className="text-white">{lineupData.model_version}</p>
                  </div>
                  <div>
                    <span className="text-slate-600 uppercase text-[9px]">Sims</span>
                    <p className="text-white">
                      {lineupData.total_simulations.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <span className="text-slate-600 uppercase text-[9px]">Tiempo</span>
                    <p className="text-white">
                      {lineupData.elapsed_seconds < 1
                        ? `${(lineupData.elapsed_seconds * 1000).toFixed(0)}ms`
                        : `${lineupData.elapsed_seconds.toFixed(2)}s`}
                    </p>
                  </div>
                </div>
              </div>
            )}
          </Card>

          {/* Uncertainty / Prediction Interval (Roadmap 0.5 — replaces vague "Confianza 82%") */}
          <Card className="flex flex-col justify-center gap-4">
            {lineupLoading ? (
              <Skeleton className="h-24 w-full" />
            ) : (
              <UncertaintyPanel
                p10={lineupData?.percentile_10}
                p90={lineupData?.percentile_90}
                p50={lineupData?.runs_scored_percentiles?.['50'] ?? lineupData?.expected_runs}
                er={lineupData?.expected_runs}
                uncertaintyLevel={lineupData?.uncertainty_level}
                stdDev={lineupData?.std_dev}
                winProbCiLow={lineupData?.win_prob_ci_low}
                winProbCiHigh={lineupData?.win_prob_ci_high}
                detail={lineupData?.model_confidence_detail}
              />
            )}
            {/* Audit 1.5 — Matchup Context sub-panel */}
            {currentLineup.length > 0 && (
              <div className="mt-3 pt-3 border-t border-slate-800">
                <MatchupContextPanel
                  lineup={currentLineup}
                  pitcherHand={(lineupData?.pitcher_hand ?? pitcher?.hand ?? 'R') as 'R' | 'L'}
                  pitcherName={lineupData?.pitcher_name ?? pitcher?.name}
                  pitcherEra={lineupData?.pitcher_era}
                />
              </div>
            )}
          </Card>
        </div>

        {/* Precision Breakdown Band (Audit 1.8 — Accuracy / MAE / Brier / n) */}
        {lineupData?.precision_30d && (
          <PrecisionBand precision={lineupData.precision_30d} />
        )}

        {/* Main Content: 2 columns 3:2 */}
        <div className="grid gap-4" style={{ gridTemplateColumns: '3fr 2fr' }}>
          {/* Left: Lineup + What-If */}
          <div className="flex flex-col gap-4">
            <Card>
              <p className="text-xs uppercase tracking-widest text-slate-500 mb-3">
                Lineup Óptimo
              </p>
              {lineupLoading ? (
                <div className="space-y-2">
                  {Array.from({ length: 9 }).map((_, i) => (
                    <Skeleton key={i} className="h-10 w-full" />
                  ))}
                </div>
              ) : currentLineup.length > 0 ? (
                <LineupTable
                  lineup={currentLineup}
                  bench={lineupData?.bench ?? []}
                  onReorder={handleReorder}
                  onWhatIfResult={handleWhatIfResult}
                  gamePk={effectiveGamePk ?? 0}
                  winProbability={lineupData?.win_probability ?? 0.5}
                />
              ) : (
                <div className="py-8 text-center text-slate-500 text-sm">
                  {lineupError
                    ? 'No se pudo cargar el lineup'
                    : 'Selecciona un partido para ver el lineup'}
                </div>
              )}
            </Card>

            <Card>
              <p className="text-xs uppercase tracking-widest text-slate-500 mb-3">
                Simulador What-If
              </p>
              <WhatIfPanel
                result={whatIfResult}
                onReset={handleReset}
                onConfirm={() => {
                  // Confirm: commit local lineup as new baseline
                  setWhatIfResult(null);
                }}
              />
            </Card>
          </div>

          {/* Right: Charts */}
          <div className="flex flex-col gap-4">
            {/* Z-Score deviation chart (Roadmap 1.1 — replaces misleading radar chart) */}
            <Card>
              {lineupLoading ? (
                <Skeleton className="h-72 w-full" />
              ) : currentLineup.length > 0 ? (
                <ZScoreChart
                  lineup={currentLineup}
                  pitcherHand={lineupData?.pitcher_hand ?? pitcher?.hand ?? 'R'}
                  pitcherName={lineupData?.pitcher_name ?? pitcher?.name ?? ''}
                />
              ) : (
                <div className="h-72 flex items-center justify-center text-slate-600 text-sm">
                  Sin datos
                </div>
              )}
            </Card>

            <Card>
              <p className="text-xs uppercase tracking-widest text-slate-500 mb-1">
                Perfil Ofensivo
              </p>
              {lineupLoading ? (
                <Skeleton className="h-72 w-full" />
              ) : currentLineup.length > 0 ? (
                <OPSChart lineup={currentLineup} />
              ) : (
                <div className="h-72 flex items-center justify-center text-slate-600 text-sm">
                  Sin datos
                </div>
              )}
            </Card>
          </div>
        </div>

        {/* Intel Briefing */}
        {lineupLoading ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <IntelBriefing
            ragText={lineupData?.rag_explanation ?? ''}
            pitcherHand={lineupData?.pitcher_hand ?? pitcher?.hand}
            modelVersion={lineupData?.model_version}
            totalSimulations={lineupData?.total_simulations}
            elapsedSeconds={lineupData?.elapsed_seconds}
          />
        )}

        {/* Footer */}
        <div className="py-4 text-center text-xs text-slate-700 font-mono">
          MLB War Room · {new Date().toLocaleDateString('es-ES')} ·
          Datos: MLB Stats API
        </div>
      </div>
    </div>
  );
}
