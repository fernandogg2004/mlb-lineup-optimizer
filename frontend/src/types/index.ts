// ─── Existing types ──────────────────────────────────────────────────────────

export interface Game {
  game_pk: number;
  home_name: string;
  away_name: string;
  home_team: string;
  away_team: string;
  game_time: string;
  venue: string;
  home_pitcher: string;
  away_pitcher: string;
  home_pitcher_hand: string;
  away_pitcher_hand: string;
}

export interface Player {
  order: number;
  player_id: number;
  name: string;
  pos: string;
  hand: string;
  avg: number;
  ops: number;
  woba: number;
  obp: number;
  iso: number;
}

export interface ModelConfidenceDetail {
  overall_pct: number;
  components: {
    /** null until backtest.py has been run (Roadmap 0.1 — no hardcoded values) */
    backtesting_accuracy_30d: number | null;
    backtesting_sample_n: number | null;
    backtesting_log_loss: number | null;
    montecarlo_stability_sigma: number;
    prediction_interval_p10_p90: number;
    data_coverage_pct: number;
    last_updated_ms: number;
  };
  degraded_threshold: number;
  optimal_threshold: number;
  /** 'backtest_file' | 'uncertainty_heuristic' */
  source?: string;
}

export interface WinProbFactor {
  factor: string;
  delta_pct: number;
  direction: 'positive' | 'negative';
}

/** 30-day model precision metrics from API (Audit 1.8 — contextualización de precisión). */
export interface Precision30d {
  accuracy_pct: number;
  mae_carreras: number;
  brier_score: number;
  brier_benchmark: number;
  n_games: number;
  /** Delta vs ELO baseline (positive = model is better). */
  vs_elo_delta: number;
}

export interface LineupData {
  game_pk: number;
  expected_runs: number;
  /** P(local gana). Complemento exacto de away_win_probability; ambos suman 1.0. */
  win_probability: number;
  /**
   * P(visitante gana) = 1 − win_probability.
   * Derivado de la misma simulación que win_probability — siempre suman 1.
   * En beisbol no hay empate: el simulador distribuye los juegos empatados 50/50.
   */
  away_win_probability?: number;
  model_confidence: number;
  total_simulations: number;
  elapsed_seconds: number;
  optimization_mode: string;
  model_version: string;
  runs_scored_percentiles: Record<string, number>;
  lineup: Player[];
  bench: Player[];
  rag_explanation: string;
  pitcher_name: string;
  pitcher_hand: string;
  pitcher_era: number | null;
  // Extended projection fields (Bug 2 — probabilistic CI)
  std_dev?: number;
  percentile_10?: number;
  percentile_25?: number;
  percentile_75?: number;
  percentile_90?: number;
  win_prob_ci_low?: number;
  win_prob_ci_high?: number;
  simulation_n?: number;
  uncertainty_level?: 'low' | 'medium' | 'high';
  // Confidence methodology breakdown (Bloque 6)
  model_confidence_detail?: ModelConfidenceDetail;
  // Win probability attribution for XAI waterfall (Bloque 8)
  win_prob_attribution?: WinProbFactor[];
  win_prob_base_pct?: number;
  // 30-day precision breakdown (Audit 1.8)
  precision_30d?: Precision30d;
}

export interface WhatIfResult {
  base_expected_runs: number;
  new_expected_runs: number;
  delta_er: number;
  base_win_probability: number;
  new_win_probability: number;
  delta_win_probability: number;
  simulation_n: number;
}

export interface IntelCard {
  type: 'advantage' | 'risk' | 'decision' | 'matchup';
  content: string;
  player?: string;
  metric?: string;
}

// ─── Post-Game Review types ───────────────────────────────────────────────────

export interface HistoricalGame {
  game_pk: number;
  home_name: string;
  away_name: string;
  home_team: string;
  away_team: string;
  game_date: string;
  final_score: string;
  home_runs_actual: number;
  away_runs_actual: number;
  home_pitcher: string;
  away_pitcher: string;
}

/** A single factor that drove a divergence decision (Bug 4 — explainability). */
export interface FeatureFactor {
  factor: string;
  shap_value: number;
  direction: 'favor_modelo' | 'favor_manager';
}

/** Full post-game report returned by /v1/report/{game_pk}. */
export interface PostGameReport {
  game_pk: number;
  game_date: string;
  matchup: string;
  game_result: boolean;
  proposed_lineup: Array<{ order: number; name: string; pos: string }>;
  actual_lineup: Array<{ order: number; name: string; pos: string; result?: string }>;
  projected_runs: number;
  actual_home_runs: number;
  actual_away_runs: number;
  win_probability_projected: number;
  model_log_loss: number;
  model_version: string;
  report_markdown: string;
  // Extended fields (Bug 2 — probabilistic CI)
  std_dev?: number;
  percentile_10?: number;
  percentile_25?: number;
  percentile_75?: number;
  percentile_90?: number;
  win_prob_ci_low?: number;
  win_prob_ci_high?: number;
  simulation_n?: number;
  uncertainty_level?: 'low' | 'medium' | 'high';
  // Divergence analysis (Bug 3 + 4)
  divergences?: DivergenceRow[];
  pitcher_hand?: string;
  /**
   * Perspective side: "home" (default, AI prediction available) or "away"
   * (only actual lineup shown, no model divergence analysis).
   */
  team_side?: 'home' | 'away';
  /** Display name of the team whose perspective is shown. */
  team_name?: string;
  /** Display name of the opposing team. */
  opponent_name?: string;
  /**
   * Origen de los datos (audit F06): 'live' = API real; 'demo' = fixtures de
   * demostración (incluye factores/SHAP ilustrativos, NO explicaciones reales
   * del modelo). La UI debe mostrar un aviso visible cuando es 'demo'.
   */
  _source?: 'live' | 'demo';
}

/** Rolling model metrics over the season (Bug 5 — replaces per-game log-loss). */
export interface RollingMetrics {
  log_loss_rolling_100: number;
  log_loss_baseline_elo: number;
  brier_score_season: number;
  brier_score_benchmark: number;
  n_games_evaluated: number;
  n_games_season: number;
  /** Percentile rank within season (0–100). */
  model_percentile?: number;
  // Temporal breakdown for Brier (Audit 2.4 — trajectory comparison)
  brier_score_30d?: number;
  brier_score_90d?: number;
  n_games_30d?: number;
  n_games_90d?: number;
  /**
   * Trend signal: positive = Brier improving (decreasing), negative = worsening.
   * Computed as brier_score_benchmark − brier_score_season (delta vs benchmark over time).
   */
  brier_trend?: number;
  /** Accuracy (proportion correct winner) over last 30 games. */
  accuracy_30d?: number;
  /** Accuracy over full season. */
  accuracy_season?: number;
}

/** Calibration bin data (Roadmap 0.3 — reliability diagram). */
export interface CalibrationBin {
  bin_lo: number;
  bin_hi: number;
  mean_predicted: number;
  observed_freq: number | null;
  count: number;
}

export interface CalibrationData {
  bins: CalibrationBin[];
  ece: number | null;
  n_games: number;
  source: string;
  generated_at?: string;
}

/** One game record for the track record page (Roadmap 0.4). */
export interface TrackRecordEntry {
  game_pk: number;
  game_date: string;
  team: string;
  side: string;
  win_probability: number | null;
  expected_runs: number | null;
  actual_home_runs: number | null;
  actual_away_runs: number | null;
  home_won: number | null;
  has_outcome: boolean;
  delta_er?: number;
  /** Out-of-sample flag: true = prediction made before game started, no data leakage. */
  is_oos?: boolean;
}

/** Extended DivergenceRow with IC90 (Roadmap 1.2). */
export interface DivergenceRow {
  slot: number;
  model_player: string;
  model_pos: string;
  manager_player: string;
  manager_pos: string;
  /** Positive = manager's choice was better; negative = model's choice was better. */
  delta_er: number;
  match: boolean;
  top_factors: FeatureFactor[];
  /** IC90 half-width: if |delta_er| < ic90_halfwidth → not statistically significant. */
  ic90_halfwidth?: number;
  /** True only when |delta_er| > ic90_halfwidth (Roadmap 1.2). */
  significant?: boolean;
}
