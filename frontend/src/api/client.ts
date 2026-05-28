import type {
  Game,
  LineupData,
  WhatIfResult,
  HistoricalGame,
  PostGameReport,
  RollingMetrics,
} from '../types';

const BASE = '';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error');
    throw new Error(`HTTP ${res.status}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ── War Room ──────────────────────────────────────────────────────────────────

export async function fetchGames(): Promise<Game[]> {
  const data = await request<{ games: Game[] }>(`${BASE}/v1/games/today`);
  return data.games;
}

export async function fetchLineup(
  game_pk: number,
  team: 'home' | 'away',
  n_sims: number = 10000
): Promise<LineupData> {
  return request<LineupData>(
    `${BASE}/v1/optimize/${game_pk}?team=${team}&n_sims=${n_sims}`
  );
}

export async function simulateWhatIf(
  game_pk: number,
  changes: Record<string, unknown>
): Promise<WhatIfResult> {
  return request<WhatIfResult>(`${BASE}/v1/optimize/${game_pk}/whatsif`, {
    method: 'POST',
    body: JSON.stringify(changes),
  });
}

// ── Post-Game Review ──────────────────────────────────────────────────────────

/** Fetch historical games for a given date (YYYY-MM-DD). */
export async function fetchHistoricalGames(date: string): Promise<HistoricalGame[]> {
  try {
    const data = await request<{ games: HistoricalGame[] }>(
      `${BASE}/v1/games/history?date=${date}`
    );
    return data.games;
  } catch {
    // Fallback to demo fixtures
    return fetchDemoFixtureGames(date);
  }
}

/** Fetch the full post-game report for a specific game.
 *  @param team  "home" (default) shows the home-team model prediction & divergences.
 *               "away" shows the visiting team's actual lineup; no model prediction.
 */
export async function fetchPostGameReport(
  game_pk: number,
  date?: string,
  team: 'home' | 'away' = 'home',
): Promise<PostGameReport> {
  const qs = new URLSearchParams();
  if (date) qs.set('date', date);
  qs.set('team', team);
  try {
    return await request<PostGameReport>(`${BASE}/v1/report/${game_pk}?${qs}`);
  } catch (e) {
    // Re-throw HTTP errors (404, 503) so the UI shows the actual problem.
    // Only fall back to demo data for network failures (API server not running).
    if (e instanceof Error && e.message.startsWith('HTTP ')) throw e;
    return fetchDemoReport(game_pk, date, team);
  }
}

/** Rolling season-level model metrics (Bug 5 — replaces single-game log-loss). */
export async function fetchRollingMetrics(): Promise<RollingMetrics> {
  try {
    return await request<RollingMetrics>(`${BASE}/v1/metrics/rolling`);
  } catch {
    return DEMO_ROLLING_METRICS;
  }
}

/** Calibration/reliability diagram data (Roadmap 0.3). */
export async function fetchCalibrationData(): Promise<import('../types').CalibrationData> {
  try {
    return await request<import('../types').CalibrationData>(`${BASE}/v1/metrics/calibration`);
  } catch {
    return {
      bins: [], ece: null, n_games: 0, source: 'unavailable',
    };
  }
}

/** Track record — all predictions with outcomes (Roadmap 0.4). */
export async function fetchTrackRecord(params?: {
  limit?: number;
  date_from?: string;
}): Promise<{
  records: import('../types').TrackRecordEntry[];
  total: number;
  metrics?: Record<string, unknown>;
  note?: string;
}> {
  const qs = new URLSearchParams();
  if (params?.limit) qs.set('limit', String(params.limit));
  if (params?.date_from) qs.set('date_from', params.date_from);
  const q = qs.toString() ? `?${qs}` : '';
  try {
    return await request(`${BASE}/v1/track-record${q}`);
  } catch {
    return { records: [], total: 0 };
  }
}

/** Backtest metrics (Roadmap 0.2). */
export async function fetchBacktestMetrics(): Promise<{
  available: boolean;
  metrics?: Record<string, number | boolean>;
  n_games?: number;
  date_range?: { from: string; to: string };
  generated_at?: string;
  command?: string;
}> {
  try {
    return await request(`${BASE}/v1/metrics/backtest`);
  } catch {
    return { available: false };
  }
}

// ── DEMO fallbacks ────────────────────────────────────────────────────────────
// All demo data shares game_pk=745400 end-to-end for full coherence (Bug 1).

const DEMO_GAME_PK = 745400;

function fetchDemoFixtureGames(date: string): HistoricalGame[] {
  return [
    {
      game_pk: DEMO_GAME_PK,
      home_name: 'New York Yankees',
      away_name: 'Toronto Blue Jays',
      home_team: 'NYY',
      away_team: 'TOR',
      game_date: date,
      final_score: '5-3',
      home_runs_actual: 5,
      away_runs_actual: 3,
      home_pitcher: 'Carlos Rodón',
      away_pitcher: 'Kevin Gausman',
    },
    {
      game_pk: 745401,
      home_name: 'Los Angeles Dodgers',
      away_name: 'San Francisco Giants',
      home_team: 'LAD',
      away_team: 'SFG',
      game_date: date,
      final_score: '7-2',
      home_runs_actual: 7,
      away_runs_actual: 2,
      home_pitcher: 'Tyler Glasnow',
      away_pitcher: 'Logan Webb',
    },
  ];
}

async function fetchDemoReport(
  game_pk: number,
  _date?: string,
  team: 'home' | 'away' = 'home',
): Promise<PostGameReport> {
  // Try loading from demo_fixtures.json (kept in public/)
  try {
    const res = await fetch('/demo_fixtures.json');
    if (res.ok) {
      const all = await res.json();
      const fixture = all.reports?.find(
        (r: PostGameReport) => r.game_pk === game_pk
      );
      if (fixture) {
        return team === 'away' ? _flipReportToAway(fixture) : fixture;
      }
    }
  } catch {
    // fall through to hardcoded demo
  }
  return team === 'away' ? _flipReportToAway(DEMO_REPORT) : DEMO_REPORT;
}

/** Flip a home-perspective report to the away-team perspective for demo mode. */
function _flipReportToAway(r: PostGameReport): PostGameReport {
  // Extract names from matchup "Away @ Home · X–Y"
  const parts = r.matchup.split('@');
  const awayName = parts[0]?.trim() ?? 'Visitante';
  const homePart = parts[1] ?? '';
  const homeName = homePart.split('·')[0]?.trim() ?? 'Local';

  return {
    ...r,
    game_result:               r.actual_away_runs > r.actual_home_runs,
    win_probability_projected: parseFloat((1 - r.win_probability_projected).toFixed(4)),
    projected_runs:            0,       // no model prediction for away
    proposed_lineup:           [],      // no AI recommendation for away
    divergences:               [],      // no divergence analysis for away
    actual_lineup:             DEMO_AWAY_LINEUP,
    team_side:                 'away',
    team_name:                 awayName,
    opponent_name:             homeName,
    report_markdown:           `> ℹ️ Sin predicción de modelo para el equipo visitante (${awayName}).\n> El sistema almacena predicciones del equipo local únicamente.\n`,
  };
}

const DEMO_AWAY_LINEUP: PostGameReport['actual_lineup'] = [
  { order: 1, name: 'George Springer',   pos: 'CF', result: '1-for-4' },
  { order: 2, name: 'Bo Bichette',       pos: 'SS', result: '0-for-4, 2K' },
  { order: 3, name: 'Vladimir Guerrero', pos: '1B', result: '1-for-3, BB' },
  { order: 4, name: 'Daulton Varsho',    pos: 'LF', result: '0-for-4' },
  { order: 5, name: 'Ernie Clement',     pos: '3B', result: '1-for-3' },
  { order: 6, name: 'Kevin Kiermaier',   pos: 'RF', result: '0-for-3' },
  { order: 7, name: 'Danny Jansen',      pos: 'C',  result: '0-for-3' },
  { order: 8, name: 'Davis Schneider',   pos: '2B', result: '0-for-3' },
  { order: 9, name: 'Whit Merrifield',   pos: 'DH', result: '0-for-3' },
];

// ── Hardcoded DEMO data (coherent across all fields) ─────────────────────────

const DEMO_ROLLING_METRICS: RollingMetrics = {
  log_loss_rolling_100: 0.389,
  log_loss_baseline_elo: 0.431,
  brier_score_season: 0.201,
  brier_score_benchmark: 0.228,
  n_games_evaluated: 47,
  n_games_season: 162,
  model_percentile: 78,
};

const DEMO_REPORT: PostGameReport = {
  game_pk: DEMO_GAME_PK,
  game_date: '2026-05-20',
  matchup: 'NYY 5  ·  TOR 3',
  game_result: true,
  proposed_lineup: [
    { order: 1, name: 'Juan Soto',        pos: 'RF' },
    { order: 2, name: 'Aaron Judge',       pos: 'CF' },
    { order: 3, name: 'Giancarlo Stanton', pos: 'DH' },
    { order: 4, name: 'Anthony Rizzo',     pos: '1B' },
    { order: 5, name: 'Gleyber Torres',    pos: '2B' },
    { order: 6, name: 'Jazz Chisholm Jr.', pos: '3B' },
    { order: 7, name: 'Austin Wells',      pos: 'C'  },
    { order: 8, name: 'Oswaldo Cabrera',   pos: 'LF' },
    { order: 9, name: 'Anthony Volpe',     pos: 'SS' },
  ],
  actual_lineup: [
    { order: 1, name: 'Aaron Judge',       pos: 'CF', result: '2-for-4, HR' },
    { order: 2, name: 'Juan Soto',         pos: 'RF', result: '1-for-3, 2 BB' },
    { order: 3, name: 'Giancarlo Stanton', pos: 'DH', result: '1-for-4, RBI' },
    { order: 4, name: 'Anthony Rizzo',     pos: '1B', result: '0-for-4' },
    { order: 5, name: 'Gleyber Torres',    pos: '2B', result: '1-for-3, BB' },
    { order: 6, name: 'Jazz Chisholm Jr.', pos: '3B', result: '0-for-4, 2K' },
    { order: 7, name: 'Austin Wells',      pos: 'C',  result: '1-for-3' },
    { order: 8, name: 'Anthony Volpe',     pos: 'SS', result: '2-for-4' },
    { order: 9, name: 'Oswaldo Cabrera',   pos: 'LF', result: '0-for-3' },
  ],
  projected_runs: 4.87,
  actual_home_runs: 5,
  actual_away_runs: 3,
  win_probability_projected: 0.59,
  model_log_loss: 0.412,
  model_version: 'v2.1.0',
  report_markdown: '',
  // Extended fields (Bug 2)
  std_dev: 1.24,
  percentile_10: 3.1,
  percentile_25: 3.9,
  percentile_75: 5.8,
  percentile_90: 7.2,
  win_prob_ci_low: 0.51,
  win_prob_ci_high: 0.67,
  simulation_n: 10000,
  uncertainty_level: 'medium',
  pitcher_hand: 'R',
  // Divergence analysis (Bug 3 + 4)
  divergences: [
    {
      slot: 1,
      model_player: 'Juan Soto',
      model_pos: 'RF',
      manager_player: 'Aaron Judge',
      manager_pos: 'CF',
      delta_er: 0.03,   // manager's choice netted +0.03 E[R] (Judge HR)
      match: false,
      top_factors: [
        { factor: 'OBP slot #1: Soto .415 vs Judge .404 (leadoff leverage)', shap_value: 0.08, direction: 'favor_modelo' },
        { factor: 'Judge: HR vs RHP esta semana (+0.11 wOBA situacional)', shap_value: 0.11, direction: 'favor_manager' },
        { factor: 'Platoon: Soto LHB vs Gausman RHP — ventaja modelo', shap_value: 0.06, direction: 'favor_modelo' },
      ],
    },
    {
      slot: 2,
      model_player: 'Aaron Judge',
      model_pos: 'CF',
      manager_player: 'Juan Soto',
      manager_pos: 'RF',
      delta_er: -0.05,
      match: false,
      top_factors: [
        { factor: 'wOBA: Judge .431 vs Soto .392 en slot #2', shap_value: 0.12, direction: 'favor_modelo' },
        { factor: 'PA/juego slot #2 = 4.6 — crítico para OBP', shap_value: 0.07, direction: 'favor_modelo' },
        { factor: 'Soto: 6 hits últimos 14 AB en Yankee Stadium', shap_value: 0.04, direction: 'favor_manager' },
      ],
    },
    {
      slot: 3, model_player: 'Giancarlo Stanton', model_pos: 'DH',
      manager_player: 'Giancarlo Stanton', manager_pos: 'DH',
      delta_er: 0.0, match: true, top_factors: [],
    },
    {
      slot: 4, model_player: 'Anthony Rizzo', model_pos: '1B',
      manager_player: 'Anthony Rizzo', manager_pos: '1B',
      delta_er: 0.0, match: true, top_factors: [],
    },
    {
      slot: 5, model_player: 'Gleyber Torres', model_pos: '2B',
      manager_player: 'Gleyber Torres', manager_pos: '2B',
      delta_er: 0.0, match: true, top_factors: [],
    },
    {
      slot: 6, model_player: 'Jazz Chisholm Jr.', model_pos: '3B',
      manager_player: 'Jazz Chisholm Jr.', manager_pos: '3B',
      delta_er: 0.0, match: true, top_factors: [],
    },
    {
      slot: 7, model_player: 'Austin Wells', model_pos: 'C',
      manager_player: 'Austin Wells', manager_pos: 'C',
      delta_er: 0.0, match: true, top_factors: [],
    },
    {
      slot: 8,
      model_player: 'Oswaldo Cabrera',
      model_pos: 'LF',
      manager_player: 'Anthony Volpe',
      manager_pos: 'SS',
      delta_er: 0.01,
      match: false,
      top_factors: [
        { factor: 'Volpe: 2-for-4 season vs Gausman (contacto)', shap_value: 0.03, direction: 'favor_manager' },
        { factor: 'Cabrera wOBA .265 vs Volpe .288 — diferencial bajo', shap_value: 0.02, direction: 'favor_manager' },
        { factor: 'Slot #8 PA/juego = 3.2 — impacto marginal', shap_value: 0.01, direction: 'neutral' as 'favor_modelo' },
      ],
    },
    {
      slot: 9,
      model_player: 'Anthony Volpe',
      model_pos: 'SS',
      manager_player: 'Oswaldo Cabrera',
      manager_pos: 'LF',
      delta_er: -0.01,
      match: false,
      top_factors: [
        { factor: 'Volpe OBP .305 > Cabrera OBP .275 — segundo leadoff', shap_value: 0.03, direction: 'favor_modelo' },
        { factor: 'Slot #9 PA/juego = 3.0 — impacto mínimo', shap_value: 0.01, direction: 'neutral' as 'favor_modelo' },
      ],
    },
  ],
};
