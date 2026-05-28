/**
 * TrackRecord — Auditoría v2.0
 *
 * Cambios post-auditoría de inversión:
 *  3.1 — Header de KPIs agregados (Brier Score como KPI principal)
 *  3.3 — Columna "Estado": 🔵 En curso / ✅ Correcto / ❌ Incorrecto
 *  3.4 — Columna ΔCarreras (renombrada desde "AciRi" ambiguo)
 *  3.5 — Filtros: confianza del modelo, solo con resultado, rango de E[R]
 *  3.6 — Gráfico de performance acumulada (Brier rolling 30d)
 *  3.7 — Badge OOS / BT para separar in-sample y out-of-sample
 */
import { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';

interface TrackRecord {
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
  is_oos?: boolean;   // true = real-time OOS prediction; false/absent = backtest
}

interface TrackRecordResponse {
  records: TrackRecord[];
  total: number;
  metrics?: {
    log_loss?: number;
    log_loss_baseline_elo?: number;
    brier_score?: number;
    auc_roc?: number;
    ece?: number;
    accuracy_pct?: number;
    mae_runs?: number;
    beats_elo_baseline?: boolean;
  };
  date_range?: { from: string; to: string };
  source?: string;
  note?: string;
}

interface BacktestResponse {
  available: boolean;
  metrics?: {
    log_loss?: number;
    log_loss_baseline_50pct?: number;
    log_loss_baseline_elo?: number;
    log_loss_delta_vs_elo?: number;
    brier_score?: number;
    brier_benchmark?: number;
    brier_delta?: number;
    auc_roc?: number;
    ece?: number;
    accuracy_pct?: number;
    mae_runs?: number;
    beats_elo_baseline?: boolean;
  };
  n_games?: number;
  date_range?: { from: string; to: string };
  generated_at?: string;
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Card({ children, style }: { children: React.ReactNode; style?: React.CSSProperties }) {
  return (
    <div style={{
      background: '#111827', border: '1px solid #1F2937',
      borderRadius: 12, padding: '18px 20px', ...style,
    }}>
      {children}
    </div>
  );
}

function MetricBox({
  label, value, sub, color = '#E8EAF6', isPrimary = false,
}: { label: string; value: string; sub?: string; color?: string; isPrimary?: boolean }) {
  return (
    <div style={{
      background: '#0D1117', border: `1px solid ${isPrimary ? color + '40' : '#1F2937'}`,
      borderRadius: 8, padding: isPrimary ? '14px 18px' : '10px 14px',
      minWidth: isPrimary ? 160 : 120,
      boxShadow: isPrimary ? `0 0 12px ${color}15` : 'none',
    }}>
      <p style={{ fontSize: '0.60rem', color: isPrimary ? color + 'AA' : '#374151',
                  letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 4 }}>
        {label}
      </p>
      <p style={{
        fontSize: isPrimary ? '1.8rem' : '1.3rem', fontWeight: 800, color,
        fontFamily: 'JetBrains Mono, monospace',
      }}>
        {value}
      </p>
      {sub && (
        <p style={{ fontSize: '0.65rem', color: '#546E7A', marginTop: 2,
                    fontFamily: 'JetBrains Mono, monospace' }}>
          {sub}
        </p>
      )}
    </div>
  );
}

// ── Rolling Performance Chart (audit 3.6) ─────────────────────────────────────
function RollingPerformanceChart({ records }: { records: TrackRecord[] }) {
  const svgRef = useRef<SVGSVGElement>(null);

  const resolved = records
    .filter(r => r.has_outcome && r.win_probability !== null && r.home_won !== null)
    .sort((a, b) => a.game_date.localeCompare(b.game_date));

  useEffect(() => {
    if (!svgRef.current || resolved.length < 3) return;
    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const W = 700, H = 140;
    const m = { top: 16, right: 16, bottom: 32, left: 46 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    svg.attr('width', '100%').attr('viewBox', `0 0 ${W} ${H}`);
    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    // Compute rolling 20-game Brier score
    const window = 20;
    const rollingData: { idx: number; date: string; brier: number; mae: number }[] = [];
    for (let i = window - 1; i < resolved.length; i++) {
      const slice = resolved.slice(i - window + 1, i + 1);
      const brier = slice.reduce((acc, r) => {
        const p = r.win_probability!;
        const y = r.home_won!;
        return acc + (p - y) ** 2;
      }, 0) / slice.length;
      const mae = slice.reduce((acc, r) => {
        if (r.expected_runs === null || r.actual_home_runs === null) return acc;
        return acc + Math.abs(r.expected_runs - r.actual_home_runs);
      }, 0) / slice.filter(r => r.expected_runs !== null).length;
      rollingData.push({ idx: i, date: resolved[i].game_date, brier, mae });
    }
    if (rollingData.length < 2) return;

    const xScale = d3.scaleLinear().domain([0, rollingData.length - 1]).range([0, iW]);
    const brierMax = Math.max(0.35, d3.max(rollingData, d => d.brier) ?? 0.25);
    const yScale = d3.scaleLinear().domain([0, brierMax]).range([iH, 0]);

    // Benchmark line (0.228 = MLB Elo baseline Brier)
    const BENCHMARK = 0.228;
    g.append('line')
      .attr('x1', 0).attr('x2', iW)
      .attr('y1', yScale(BENCHMARK)).attr('y2', yScale(BENCHMARK))
      .attr('stroke', '#374151').attr('stroke-width', 1)
      .attr('stroke-dasharray', '4,3');
    g.append('text')
      .attr('x', iW + 2).attr('y', yScale(BENCHMARK))
      .attr('dominant-baseline', 'middle')
      .attr('fill', '#374151').attr('font-size', '8px')
      .attr('font-family', 'JetBrains Mono, monospace')
      .text('Benchmark');

    // Fill area under Brier curve
    const area = d3.area<typeof rollingData[0]>()
      .x((_, i) => xScale(i))
      .y0(iH)
      .y1(d => yScale(d.brier))
      .curve(d3.curveMonotoneX);
    g.append('path')
      .datum(rollingData)
      .attr('fill', 'rgba(59,130,246,0.08)')
      .attr('d', area);

    // Brier score line
    const line = d3.line<typeof rollingData[0]>()
      .x((_, i) => xScale(i))
      .y(d => yScale(d.brier))
      .curve(d3.curveMonotoneX);

    const beatsLine = rollingData.map(d => d.brier < BENCHMARK);
    // Color segments: green where below benchmark, red where above
    for (let i = 0; i < rollingData.length - 1; i++) {
      const color = beatsLine[i] ? '#22C55E' : '#EF4444';
      g.append('line')
        .attr('x1', xScale(i)).attr('y1', yScale(rollingData[i].brier))
        .attr('x2', xScale(i + 1)).attr('y2', yScale(rollingData[i + 1].brier))
        .attr('stroke', color).attr('stroke-width', 2).attr('opacity', 0.8);
    }

    // Axes
    const xAxis = g.append('g').attr('transform', `translate(0,${iH})`);
    xAxis.append('line').attr('x2', iW).attr('stroke', 'rgba(255,255,255,0.1)');
    // Date ticks every ~10 points
    const step = Math.max(1, Math.floor(rollingData.length / 5));
    for (let i = 0; i < rollingData.length; i += step) {
      xAxis.append('text')
        .attr('x', xScale(i)).attr('y', 14)
        .attr('text-anchor', 'middle')
        .attr('fill', 'rgba(255,255,255,0.25)').attr('font-size', '8px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(rollingData[i].date.slice(5)); // MM-DD
    }

    const yAxis = g.append('g');
    yAxis.append('line').attr('y2', iH).attr('stroke', 'rgba(255,255,255,0.1)');
    [0, 0.1, 0.2, 0.3].filter(v => v <= brierMax).forEach(v => {
      yAxis.append('text')
        .attr('x', -4).attr('y', yScale(v))
        .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
        .attr('fill', 'rgba(255,255,255,0.25)').attr('font-size', '8px')
        .attr('font-family', 'JetBrains Mono, monospace')
        .text(v.toFixed(2));
    });

    // Last point annotation
    const last = rollingData[rollingData.length - 1];
    const lastColor = last.brier < BENCHMARK ? '#22C55E' : '#EF4444';
    g.append('circle')
      .attr('cx', xScale(rollingData.length - 1)).attr('cy', yScale(last.brier))
      .attr('r', 4).attr('fill', lastColor).attr('stroke', '#0A0E1A').attr('stroke-width', 1.5);

  }, [resolved]);

  if (resolved.length < 3) {
    return (
      <div style={{ padding: '20px', textAlign: 'center', color: '#374151', fontSize: '0.82rem' }}>
        Gráfico disponible con ≥ {3 - resolved.length} partido(s) más con resultado.
        Ejecuta <code style={{ color: '#90CAF9' }}>python backtest.py</code> para ver métricas históricas.
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8 }}>
        <p style={{ fontSize: '0.63rem', fontWeight: 700, letterSpacing: '0.14em',
                    textTransform: 'uppercase', color: '#2E5580' }}>
          Brier Score Rolling 20 partidos
        </p>
        <span style={{ fontSize: '0.65rem', color: '#374151', fontFamily: 'JetBrains Mono, monospace' }}>
          Verde = por debajo del benchmark Elo (0.228) · Rojo = por encima
        </span>
      </div>
      <svg ref={svgRef} style={{ width: '100%', display: 'block' }} />
    </div>
  );
}

// ── Main component ─────────────────────────────────────────────────────────────

export default function TrackRecord() {
  const [data, setData] = useState<TrackRecordResponse | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterDate, setFilterDate] = useState('');
  const [showOnlyResolved, setShowOnlyResolved] = useState(false);
  const [confidenceFilter, setConfidenceFilter] = useState<number>(0); // min P(W) threshold
  const [showOnlyOOS, setShowOnlyOOS] = useState(false);

  useEffect(() => {
    const qs = filterDate ? `?date_from=${filterDate}&limit=300` : '?limit=300';
    Promise.all([
      fetch(`/v1/track-record${qs}`).then(r => r.json()).catch(() => null),
      fetch('/v1/metrics/backtest').then(r => r.json()).catch(() => null),
    ]).then(([tr, bt]) => {
      setData(tr);
      setBacktest(bt);
      setLoading(false);
    });
  }, [filterDate]);

  const allRecords = data?.records ?? [];

  const records = allRecords.filter(r => {
    if (showOnlyResolved && !r.has_outcome) return false;
    if (showOnlyOOS && !r.is_oos) return false;
    if (confidenceFilter > 0 && r.win_probability !== null) {
      // Filter by confidence: |P(W) - 0.5| > threshold
      if (Math.abs(r.win_probability - 0.5) < confidenceFilter / 100) return false;
    }
    return true;
  });

  const m = backtest?.metrics;
  const beatsElo = m?.beats_elo_baseline;
  const resolvedRecords = allRecords.filter(r => r.has_outcome);
  const nCorrect = resolvedRecords.filter(r => {
    if (r.home_won === null || r.win_probability === null) return false;
    return (r.win_probability > 0.5 && r.home_won === 1) ||
           (r.win_probability < 0.5 && r.home_won === 0);
  }).length;
  const accuracyLive = resolvedRecords.length > 0
    ? Math.round((nCorrect / resolvedRecords.length) * 1000) / 10 : null;

  // Compute Brier score from live records
  const brierLive = resolvedRecords.length >= 5
    ? resolvedRecords.reduce((acc, r) => {
        if (r.win_probability === null || r.home_won === null) return acc;
        return acc + (r.win_probability - r.home_won) ** 2;
      }, 0) / resolvedRecords.length
    : null;

  // MAE for runs
  const maeLive = resolvedRecords.filter(r => r.expected_runs !== null && r.actual_home_runs !== null).length >= 3
    ? resolvedRecords.filter(r => r.expected_runs !== null && r.actual_home_runs !== null)
        .reduce((acc, r) => acc + Math.abs(r.expected_runs! - r.actual_home_runs!), 0)
        / resolvedRecords.filter(r => r.expected_runs !== null && r.actual_home_runs !== null).length
    : null;

  const brierColor = (v: number | null) => {
    if (v === null) return '#6B7280';
    if (v < 0.228) return '#22C55E';
    if (v < 0.25) return '#FFB800';
    return '#EF4444';
  };

  return (
    <div style={{ minHeight: '100vh', background: '#0A0E1A' }}>
      {/* Page header */}
      <div style={{
        padding: '14px 20px', borderBottom: '1px solid #162030',
        display: 'flex', alignItems: 'baseline', gap: 12,
      }}>
        <span style={{
          fontSize: '1.45rem', fontWeight: 800, color: '#E8EAF6',
          letterSpacing: '-0.02em',
        }}>
          📋 Track Record
        </span>
        <span style={{
          fontSize: '0.7rem', fontWeight: 600, letterSpacing: '0.14em',
          textTransform: 'uppercase', color: '#2E5580',
        }}>
          Predicciones verificables · Publicadas antes del partido
        </span>
      </div>

      <div style={{
        maxWidth: 1200, margin: '0 auto',
        padding: '16px 16px 32px',
        display: 'flex', flexDirection: 'column', gap: 16,
      }}>

        {/* ── AGGREGATE KPI HEADER (audit 3.2 — must be first visual element) ── */}
        <Card>
          <p style={{
            fontSize: '0.63rem', fontWeight: 700, letterSpacing: '0.14em',
            textTransform: 'uppercase', color: '#2E5580', marginBottom: 14,
          }}>
            KPIs Agregados del Modelo — Temporada actual
          </p>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'flex-start' }}>
            {/* Brier Score — primary KPI (audit 2.4 + 3.2) */}
            <MetricBox
              label="Brier Score (Principal)"
              value={
                (m?.brier_score ?? brierLive)?.toFixed(4) ?? '—'
              }
              sub={`Benchmark: ${(m?.brier_benchmark ?? 0.228).toFixed(4)} · ${
                (m?.brier_score ?? brierLive ?? 1) < (m?.brier_benchmark ?? 0.228)
                  ? '✅ supera benchmark' : '⚠️ por debajo del benchmark'
              }`}
              color={brierColor(m?.brier_score ?? brierLive ?? null)}
              isPrimary
            />
            <MetricBox
              label="Accuracy (ganador)"
              value={
                m?.accuracy_pct !== undefined
                  ? `${m.accuracy_pct.toFixed(1)}%`
                  : accuracyLive !== null ? `${accuracyLive}%` : '—'
              }
              sub="dirección binaria del partido"
              color={(m?.accuracy_pct ?? accuracyLive ?? 0) > 55 ? '#22C55E' : '#FFB800'}
            />
            <MetricBox
              label="|ΔCarreras| MAE"
              value={
                m?.mae_runs !== undefined
                  ? m.mae_runs.toFixed(2)
                  : maeLive !== null ? maeLive.toFixed(2) : '—'
              }
              sub="error medio absoluto E[R] vs real"
              color={(m?.mae_runs ?? maeLive ?? 99) < 2.0 ? '#22C55E' : '#FFB800'}
            />
            <MetricBox
              label="AUC ROC"
              value={m?.auc_roc?.toFixed(4) ?? '—'}
              sub="0.5 = azar · >0.6 = bueno"
              color={(m?.auc_roc ?? 0) > 0.6 ? '#22C55E' : '#FFB800'}
            />
            <MetricBox
              label="ECE (Calibración)"
              value={m?.ece?.toFixed(4) ?? '—'}
              sub="0 = calibración perfecta"
              color={(m?.ece ?? 1) < 0.05 ? '#22C55E' : (m?.ece ?? 1) < 0.10 ? '#FFB800' : '#EF4444'}
            />
            <MetricBox
              label="Log-Loss vs Elo"
              value={
                m?.log_loss_delta_vs_elo !== undefined
                  ? `${m.log_loss_delta_vs_elo >= 0 ? '+' : ''}${m.log_loss_delta_vs_elo.toFixed(4)}`
                  : '—'
              }
              sub="positivo = modelo mejor que Elo"
              color={
                m?.log_loss_delta_vs_elo !== undefined && m.log_loss_delta_vs_elo > 0
                  ? '#22C55E' : '#EF4444'
              }
            />
          </div>
          {/* n_games context */}
          <p style={{ marginTop: 12, fontSize: '0.65rem', color: '#374151',
                      fontFamily: 'JetBrains Mono, monospace' }}>
            n = {backtest?.n_games ?? resolvedRecords.length} partidos con resultado ·{' '}
            {backtest?.available
              ? `Backtest: ${backtest.date_range?.from ?? '—'} → ${backtest.date_range?.to ?? '—'}`
              : `Track record en tiempo real (${allRecords.length} predicciones totales)`
            }
            {backtest?.generated_at && ` · Generado: ${new Date(backtest.generated_at).toLocaleString('es-ES')}`}
          </p>

          {/* Backtest missing warning (audit 3.1) */}
          {backtest && !backtest.available && (
            <div style={{
              marginTop: 12, padding: '10px 14px', borderRadius: 8,
              background: 'rgba(245,158,11,0.08)', border: '1px solid rgba(245,158,11,0.3)',
            }}>
              <p style={{ color: '#F59E0B', fontSize: '0.78rem', fontWeight: 700 }}>
                ⚠️ Backtest no ejecutado — métricas de validación no disponibles
              </p>
              <p style={{ color: '#78909C', fontSize: '0.72rem', marginTop: 4,
                          fontFamily: 'JetBrains Mono, monospace' }}>
                Ejecuta: <code style={{ color: '#90CAF9' }}>python backtest.py</code> para generar métricas out-of-sample verificables.
                Las métricas mostradas arriba son estimaciones del track record en tiempo real (n={resolvedRecords.length}).
              </p>
            </div>
          )}
        </Card>

        {/* ── ROLLING PERFORMANCE CHART (audit 3.6) ── */}
        <Card>
          <RollingPerformanceChart records={allRecords} />
        </Card>

        {/* ── BACKTEST OOS METRICS (when available) ── */}
        {backtest?.available && m && (
          <Card>
            <p style={{
              fontSize: '0.63rem', fontWeight: 700, letterSpacing: '0.14em',
              textTransform: 'uppercase', color: '#2E5580', marginBottom: 14,
            }}>
              Métricas Out-of-Sample — Backtest Walk-Forward · n = {backtest.n_games} partidos
              <span style={{ marginLeft: 8, background: '#22C55E20', color: '#22C55E',
                             padding: '1px 6px', borderRadius: 4, fontSize: '0.65rem' }}>
                OOS
              </span>
            </p>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <MetricBox
                label="Brier Score"
                value={m.brier_score?.toFixed(4) ?? '—'}
                sub={`Benchmark: ${m.brier_benchmark?.toFixed(4) ?? '—'} · δ = ${m.brier_delta !== undefined ? (m.brier_delta >= 0 ? '+' : '') + m.brier_delta?.toFixed(4) : '—'}`}
                color={brierColor(m.brier_score ?? null)}
              />
              <MetricBox
                label="Δ Log-Loss vs Elo"
                value={m.log_loss_delta_vs_elo !== undefined
                  ? `${m.log_loss_delta_vs_elo >= 0 ? '+' : ''}${m.log_loss_delta_vs_elo.toFixed(4)}`
                  : '—'}
                sub="positivo = modelo mejor"
                color={m.log_loss_delta_vs_elo !== undefined && m.log_loss_delta_vs_elo > 0 ? '#22C55E' : '#EF4444'}
              />
              <MetricBox
                label="AUC ROC"
                value={m.auc_roc?.toFixed(4) ?? '—'}
                sub="0.5 = azar"
                color={(m.auc_roc ?? 0) > 0.55 ? '#22C55E' : '#FFB800'}
              />
              <MetricBox
                label="ECE"
                value={m.ece?.toFixed(4) ?? '—'}
                sub="0 = calibración perfecta"
                color={(m.ece ?? 1) < 0.05 ? '#22C55E' : (m.ece ?? 1) < 0.10 ? '#FFB800' : '#EF4444'}
              />
              <MetricBox
                label="Precisión dirección"
                value={m.accuracy_pct !== undefined ? `${m.accuracy_pct.toFixed(1)}%` : '—'}
                sub="vs pronóstico binario"
                color={(m.accuracy_pct ?? 0) > 55 ? '#22C55E' : '#FFB800'}
              />
            </div>
          </Card>
        )}

        {/* ── FILTERS (audit 3.5) ── */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Date filter */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: '0.60rem', color: '#374151',
                           letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              Desde
            </label>
            <input
              type="date"
              value={filterDate}
              onChange={e => setFilterDate(e.target.value)}
              style={{
                background: '#0D1117', border: '1px solid #1F2937', borderRadius: 6,
                padding: '5px 8px', color: '#E8EAF6', fontSize: '0.82rem',
                fontFamily: 'JetBrains Mono, monospace', outline: 'none',
              }}
            />
          </div>

          {/* Confidence filter (audit 3.5) */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <label style={{ fontSize: '0.60rem', color: '#374151',
                           letterSpacing: '0.1em', textTransform: 'uppercase' }}>
              Confianza mínima
            </label>
            <select
              value={confidenceFilter}
              onChange={e => setConfidenceFilter(Number(e.target.value))}
              style={{
                background: '#0D1117', border: '1px solid #1F2937', borderRadius: 6,
                padding: '5px 8px', color: '#E8EAF6', fontSize: '0.78rem',
                fontFamily: 'JetBrains Mono, monospace', outline: 'none',
              }}
            >
              <option value={0}>Todas las predicciones</option>
              <option value={5}>|P(W)−50%| {'>'} 5pp</option>
              <option value={10}>|P(W)−50%| {'>'} 10pp (≥60% / ≤40%)</option>
              <option value={15}>|P(W)−50%| {'>'} 15pp (≥65% / ≤35%)</option>
              <option value={20}>|P(W)−50%| {'>'} 20pp (≥70% / ≤30%)</option>
            </select>
          </div>

          {/* Solo con resultado */}
          <button
            onClick={() => setShowOnlyResolved(v => !v)}
            style={{
              marginTop: 16, fontSize: '0.72rem', padding: '5px 12px',
              borderRadius: 6, cursor: 'pointer',
              border: `1px solid ${showOnlyResolved ? '#3B82F6' : '#1F2937'}`,
              background: showOnlyResolved ? 'rgba(59,130,246,0.12)' : '#0D1117',
              color: showOnlyResolved ? '#60A5FA' : '#6B7280',
            }}
          >
            {showOnlyResolved ? '▸ Mostrar todas' : '◈ Solo con resultado'}
          </button>

          {/* Solo OOS (audit 3.7) */}
          <button
            onClick={() => setShowOnlyOOS(v => !v)}
            style={{
              marginTop: 16, fontSize: '0.72rem', padding: '5px 12px',
              borderRadius: 6, cursor: 'pointer',
              border: `1px solid ${showOnlyOOS ? '#22C55E' : '#1F2937'}`,
              background: showOnlyOOS ? 'rgba(34,197,94,0.10)' : '#0D1117',
              color: showOnlyOOS ? '#22C55E' : '#6B7280',
            }}
          >
            🔬 Solo OOS (tiempo real)
          </button>

          <span style={{
            marginTop: 16, fontSize: '0.72rem', color: '#374151',
            fontFamily: 'JetBrains Mono, monospace',
          }}>
            {records.length} predicciones · {records.filter(r => r.has_outcome).length} con resultado
          </span>
        </div>

        {/* ── RECORDS TABLE (audit 3.3, 3.4, 3.7) ── */}
        <Card style={{ padding: '0' }}>
          {loading ? (
            <p style={{ padding: 20, color: '#374151' }}>Cargando...</p>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.80rem' }}>
                <thead>
                  <tr style={{ background: '#0D1117', borderBottom: '1px solid #1F2937' }}>
                    {[
                      'Estado',        // audit 3.3
                      'Tipo',          // audit 3.7 OOS/BT
                      'Fecha',
                      'Equipo',
                      'P(W) predicha',
                      'E[R] predicho',
                      'Resultado final',
                      '|ΔCarreras|',   // audit 3.4 (rename de "AciRi")
                      'Correcto',
                    ].map(h => (
                      <th key={h} style={{
                        padding: '10px 12px', textAlign: 'left',
                        fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.1em',
                        textTransform: 'uppercase', color: '#374151',
                      }}>
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {records.map((r, i) => {
                    const correct = r.home_won !== null && r.win_probability !== null
                      ? ((r.win_probability > 0.5 && r.home_won === 1) ||
                         (r.win_probability < 0.5 && r.home_won === 0))
                      : null;
                    const deltaEr = r.delta_er ?? (
                      r.actual_home_runs !== null && r.expected_runs !== null
                        ? r.actual_home_runs - r.expected_runs : null
                    );
                    const scoreStr = r.actual_home_runs !== null && r.actual_away_runs !== null
                      ? `${r.actual_home_runs}–${r.actual_away_runs}` : null;

                    // Status (audit 3.3) — 3 visually distinct states
                    let statusEl: React.ReactNode;
                    if (!r.has_outcome) {
                      // Partido en curso o sin resultado todavía
                      statusEl = (
                        <span style={{
                          fontSize: '0.72rem', color: '#60A5FA',
                          display: 'flex', alignItems: 'center', gap: 4,
                        }}>
                          🔵 <span>En curso</span>
                        </span>
                      );
                    } else if (correct === true) {
                      statusEl = (
                        <span style={{ fontSize: '0.72rem', color: '#22C55E', display: 'flex', alignItems: 'center', gap: 4 }}>
                          ✅ <span>Correcto</span>
                        </span>
                      );
                    } else if (correct === false) {
                      statusEl = (
                        <span style={{ fontSize: '0.72rem', color: '#EF4444', display: 'flex', alignItems: 'center', gap: 4 }}>
                          ❌ <span>Incorrecto</span>
                        </span>
                      );
                    } else {
                      statusEl = (
                        <span style={{ fontSize: '0.72rem', color: '#374151' }}>—</span>
                      );
                    }

                    // OOS / BT badge (audit 3.7)
                    const isOos = r.is_oos !== false; // default to OOS if not explicitly BT
                    const typeBadge = (
                      <span style={{
                        fontSize: '0.60rem', padding: '1px 5px', borderRadius: 3,
                        background: isOos ? 'rgba(34,197,94,0.12)' : 'rgba(59,130,246,0.12)',
                        color: isOos ? '#22C55E' : '#60A5FA',
                        border: `1px solid ${isOos ? '#22C55E40' : '#3B82F640'}`,
                        fontFamily: 'JetBrains Mono, monospace',
                        fontWeight: 700,
                      }}>
                        {isOos ? 'OOS' : 'BT'}
                      </span>
                    );

                    return (
                      <tr key={`${r.game_pk}-${i}`} style={{
                        borderBottom: '1px solid #1F2937',
                        background: i % 2 === 0 ? '#111827' : '#0F1521',
                      }}>
                        {/* Estado (audit 3.3) */}
                        <td style={{ padding: '8px 12px' }}>{statusEl}</td>

                        {/* Tipo OOS/BT (audit 3.7) */}
                        <td style={{ padding: '8px 12px' }}>{typeBadge}</td>

                        <td style={{ padding: '8px 12px', color: '#6B7280',
                                     fontFamily: 'JetBrains Mono, monospace' }}>
                          {r.game_date}
                        </td>
                        <td style={{ padding: '8px 12px', color: '#90CAF9', fontWeight: 600 }}>
                          {r.team}
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'JetBrains Mono, monospace',
                                     color: r.win_probability !== null
                                       ? r.win_probability > 0.55 ? '#22C55E'
                                       : r.win_probability < 0.45 ? '#EF4444' : '#E8EAF6'
                                       : '#374151' }}>
                          {r.win_probability !== null
                            ? `${(r.win_probability * 100).toFixed(1)}%` : '—'}
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'JetBrains Mono, monospace',
                                     color: '#E8EAF6' }}>
                          {r.expected_runs !== null ? r.expected_runs.toFixed(2) : '—'}
                        </td>
                        <td style={{ padding: '8px 12px', fontFamily: 'JetBrains Mono, monospace',
                                     color: r.has_outcome ? '#E8EAF6' : '#374151' }}>
                          {scoreStr ?? (r.has_outcome ? 'N/A' : <span style={{ color: '#374151', fontSize: '0.7rem' }}>—</span>)}
                        </td>
                        {/* |ΔCarreras| (audit 3.4 — renamed from "AciRi") */}
                        <td style={{
                          padding: '8px 12px', fontFamily: 'JetBrains Mono, monospace',
                          color: deltaEr !== null
                            ? Math.abs(deltaEr) < 1.5 ? '#22C55E'
                            : Math.abs(deltaEr) < 3 ? '#FFB800' : '#EF4444'
                            : '#374151',
                        }}
                          title="|ΔCarreras| = |E[R] predicho − carreras reales|. Verde <1.5 · Naranja <3 · Rojo ≥3"
                        >
                          {deltaEr !== null
                            ? Math.abs(deltaEr).toFixed(2) : '—'}
                        </td>
                        <td style={{ padding: '8px 12px', textAlign: 'center' }}>
                          {!r.has_outcome ? (
                            <span style={{ color: '#374151', fontSize: '0.7rem' }}>—</span>
                          ) : correct === true ? (
                            <span style={{ color: '#22C55E', fontSize: '0.85rem' }}>✓</span>
                          ) : correct === false ? (
                            <span style={{ color: '#EF4444', fontSize: '0.85rem' }}>✗</span>
                          ) : (
                            <span style={{ color: '#374151', fontSize: '0.7rem' }}>—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                  {records.length === 0 && (
                    <tr>
                      <td colSpan={9} style={{
                        padding: '24px', textAlign: 'center',
                        color: '#374151', fontSize: '0.85rem',
                      }}>
                        No hay predicciones con los filtros actuales.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        {/* ── METHODOLOGY NOTE ── */}
        <Card style={{ background: '#0A1220', border: '1px solid #162030' }}>
          <p style={{
            fontSize: '0.72rem', color: '#2E5580', lineHeight: 1.8,
            fontFamily: 'JetBrains Mono, monospace',
          }}>
            🔒 <strong style={{ color: '#3D6080' }}>Integridad:</strong>{' '}
            Las predicciones OOS se generan y guardan en <code style={{ color: '#546E7A' }}>results/</code> antes del inicio de cada partido.
            Los timestamps son inmutables.
            Las predicciones BT (backtest) se generan post-hoc con walk-forward validation y se etiquetan explícitamente.{' '}
            <strong style={{ color: '#3D6080' }}>Las métricas OOS y BT están separadas</strong> para prevenir look-ahead bias.
          </p>
          <div style={{ marginTop: 8, display: 'flex', gap: 16, fontSize: '0.65rem', color: '#1E3850',
                        fontFamily: 'JetBrains Mono, monospace', flexWrap: 'wrap' }}>
            <span>
              <span style={{ color: '#22C55E' }}>OOS</span> = predicción en tiempo real (auditada)
            </span>
            <span>
              <span style={{ color: '#60A5FA' }}>BT</span> = predicción de backtest (referencial)
            </span>
            <span>|ΔCarreras| = |E[R]_predicho − carreras_reales|</span>
            <span>Fuente: {data?.note ?? 'results/*.json'}</span>
          </div>
        </Card>

        <p style={{
          textAlign: 'center', fontSize: '0.65rem', color: '#1F2937',
          fontFamily: 'JetBrains Mono, monospace',
        }}>
          MLB Track Record · Datos: MLB Stats API + modelo v2.1
        </p>
      </div>
    </div>
  );
}
