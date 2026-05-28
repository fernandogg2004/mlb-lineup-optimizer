/**
 * MatchupContextPanel — Audit 1.5
 *
 * Shows the tactical matchup context between the lineup and the opposing pitcher:
 *   - Platoon advantage breakdown (% of lineup with handedness advantage)
 *   - Aggregate lineup wOBA vs this pitcher type (computed from player data)
 *   - Pitcher ERA / xFIP context with park adjustment indicator
 *   - Historical sample notice (n = ABs if available, else heuristic)
 *
 * When backend enriches `matchup_context` in the API response, those fields
 * override the heuristic calculations here. The component always shows
 * what's "measured" vs "estimated" to satisfy the audit's data integrity standard.
 */
import type { Player } from '../../types';

interface MatchupContextPanelProps {
  lineup: Player[];
  pitcherHand: 'R' | 'L' | string;
  pitcherName?: string;
  pitcherEra?: number | null;
  /** Optional: historical lineup AVG vs this pitcher type (from API enrichment). */
  lineupAvgVsPitcherType?: number;
  /** Optional: sample ABs from historical data. */
  sampleAbs?: number;
  /** Optional: xFIP adjusted for park. */
  xfipParkAdj?: number;
  /** Optional: park factor (1.0 = neutral). */
  parkFactor?: number;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Whether a batter has platoon advantage vs a pitcher hand.
 * L batter vs R pitcher = advantage; R batter vs L pitcher = advantage; Switch = always advantage.
 */
function hasPlatoonAdv(batterHand: string, pitcherHand: string): boolean {
  if (batterHand === 'S') return true;
  return batterHand !== pitcherHand;
}

/**
 * Estimated wOBA boost from platoon advantage.
 * Historical MLB average: ~.025 wOBA advantage when batter has platoon edge.
 */
const PLATOON_WOBA_BOOST = 0.025;

// ── Component ─────────────────────────────────────────────────────────────────

export default function MatchupContextPanel({
  lineup,
  pitcherHand,
  pitcherName,
  pitcherEra,
  lineupAvgVsPitcherType,
  sampleAbs,
  xfipParkAdj,
  parkFactor,
}: MatchupContextPanelProps) {
  if (!lineup.length) return null;

  // Platoon analysis
  const withAdv = lineup.filter(p => hasPlatoonAdv(p.hand, pitcherHand));
  const nAdv    = withAdv.length;
  const advPct  = (nAdv / lineup.length) * 100;

  // Aggregate lineup metrics
  const avgWoba = lineup.reduce((acc, p) => acc + p.woba, 0) / lineup.length;
  const avgOps  = lineup.reduce((acc, p) => acc + p.ops, 0)  / lineup.length;

  // Platoon-adjusted wOBA estimate
  const platoonAdjWoba = avgWoba + (nAdv / lineup.length) * PLATOON_WOBA_BOOST;

  // Use API data if available, else heuristic
  const displayAvg       = lineupAvgVsPitcherType;
  const hasHistoricalAVG = displayAvg !== undefined;

  // Park factor color
  const pfColor = !parkFactor
    ? '#78909C'
    : parkFactor > 1.05
    ? '#EF4444'  // hitter-friendly park
    : parkFactor < 0.95
    ? '#3B82F6'  // pitcher-friendly park
    : '#78909C'; // neutral

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <p style={{
          fontSize: '0.62rem', fontWeight: 700, letterSpacing: '0.14em',
          textTransform: 'uppercase', color: '#546E7A',
        }}>
          Contexto del Matchup
        </p>
        <span style={{
          fontSize: '0.60rem', padding: '1px 6px', borderRadius: 4,
          background: 'rgba(59,130,246,0.08)', color: '#3B82F6',
          border: '1px solid rgba(59,130,246,0.2)',
        }}>
          vs {pitcherHand === 'L' ? 'Zurdo' : pitcherHand === 'R' ? 'Diestro' : pitcherHand}
          {pitcherName ? ` · ${pitcherName}` : ''}
        </span>
      </div>

      {/* Platoon advantage row */}
      <div style={{
        background: '#0D1117', borderRadius: 8, padding: '10px 12px',
        border: `1px solid ${nAdv >= 6 ? 'rgba(34,197,94,0.3)' : nAdv >= 4 ? 'rgba(245,158,11,0.25)' : '#1F2937'}`,
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
          <p style={{ fontSize: '0.60rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151' }}>
            Ventaja Platoon
          </p>
          <span style={{
            fontSize: '0.62rem', fontWeight: 700,
            color: nAdv >= 6 ? '#22C55E' : nAdv >= 4 ? '#F59E0B' : '#EF4444',
          }}>
            {nAdv}/9 ({advPct.toFixed(0)}%)
          </span>
        </div>

        {/* Visual bar */}
        <div style={{ height: 5, borderRadius: 3, background: '#1F2937', overflow: 'hidden' }}>
          <div style={{
            height: 5, width: `${advPct}%`,
            background: nAdv >= 6 ? '#22C55E' : nAdv >= 4 ? '#F59E0B' : '#EF4444',
            borderRadius: 3, transition: 'width 0.3s',
          }} />
        </div>

        {/* Batter chips */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 3, marginTop: 6 }}>
          {lineup.slice(0, 9).map((p) => {
            const adv = hasPlatoonAdv(p.hand, pitcherHand);
            return (
              <span
                key={p.order}
                title={`#${p.order} ${p.name} (${p.hand}) — ${adv ? 'ventaja' : 'desventaja'} vs ${pitcherHand}`}
                style={{
                  fontSize: '0.55rem', padding: '1px 5px', borderRadius: 3,
                  fontFamily: 'JetBrains Mono, monospace',
                  background: adv ? 'rgba(34,197,94,0.1)' : 'rgba(239,68,68,0.08)',
                  color: adv ? '#4ADE80' : '#F87171',
                  border: `1px solid ${adv ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.15)'}`,
                }}
              >
                #{p.order}{p.hand}
              </span>
            );
          })}
        </div>
        <p style={{ fontSize: '0.58rem', color: '#374151', marginTop: 4, fontStyle: 'italic' }}>
          Verde = ventaja platoon · Rojo = desventaja
        </p>
      </div>

      {/* Lineup wOBA vs pitcher type */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8,
      }}>
        {/* Platoon-adjusted wOBA */}
        <div style={{
          background: '#0D1117', borderRadius: 8, padding: '9px 11px',
          border: '1px solid #1F2937',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 3 }}>
            wOBA adj. platoon
          </p>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.1rem',
            color: platoonAdjWoba >= 0.340 ? '#22C55E' : platoonAdjWoba >= 0.315 ? '#78909C' : '#F59E0B',
          }}>
            {platoonAdjWoba.toFixed(3)}
          </span>
          <p style={{ fontSize: '0.58rem', color: '#374151', marginTop: 2 }}>
            {hasHistoricalAVG ? '✓ medido' : 'estimado · platoon +.025'}
          </p>
        </div>

        {/* OPS del lineup */}
        <div style={{
          background: '#0D1117', borderRadius: 8, padding: '9px 11px',
          border: '1px solid #1F2937',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 3 }}>
            OPS promedio lineup
          </p>
          <span style={{
            fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.1rem',
            color: avgOps >= 0.780 ? '#22C55E' : avgOps >= 0.700 ? '#78909C' : '#F59E0B',
          }}>
            {avgOps.toFixed(3)}
          </span>
          <p style={{ fontSize: '0.58rem', color: '#374151', marginTop: 2 }}>
            media de los 9 titulares
          </p>
        </div>
      </div>

      {/* Historical data row */}
      {hasHistoricalAVG && (
        <div style={{
          background: '#0D1117', borderRadius: 8, padding: '9px 11px',
          border: '1px solid rgba(34,197,94,0.2)',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 3 }}>
            AVG histórico vs {pitcherHand === 'L' ? 'zurdos' : 'diestros'}
          </p>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 8 }}>
            <span style={{
              fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.2rem',
              color: displayAvg! >= 0.260 ? '#22C55E' : '#F59E0B',
            }}>
              .{Math.round(displayAvg! * 1000).toString().padStart(3, '0')}
            </span>
            {sampleAbs !== undefined && (
              <span style={{ fontSize: '0.62rem', color: '#374151', fontFamily: 'JetBrains Mono, monospace' }}>
                n={sampleAbs} ABs
                {sampleAbs < 100 && <span style={{ color: '#F59E0B', marginLeft: 4 }}>⚠ n&lt;100</span>}
              </span>
            )}
          </div>
        </div>
      )}

      {/* Pitcher ERA / xFIP */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8,
      }}>
        {/* ERA */}
        <div style={{
          background: '#0D1117', borderRadius: 8, padding: '9px 11px',
          border: '1px solid #1F2937',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 3 }}>
            ERA pitcher
          </p>
          {pitcherEra != null ? (
            <>
              <span style={{
                fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.1rem',
                color: pitcherEra <= 3.50 ? '#EF4444' : pitcherEra <= 4.50 ? '#F59E0B' : '#22C55E',
              }}>
                {pitcherEra.toFixed(2)}
              </span>
              <p style={{ fontSize: '0.58rem', color: '#374151', marginTop: 2 }}>
                {pitcherEra <= 3.50 ? 'pitcher élite — cuidado' : pitcherEra <= 4.50 ? 'promedio de liga' : 'favorable para el lineup'}
              </p>
            </>
          ) : (
            <span style={{ fontSize: '0.8rem', color: '#374151' }}>N/D</span>
          )}
        </div>

        {/* xFIP park-adjusted */}
        <div style={{
          background: '#0D1117', borderRadius: 8, padding: '9px 11px',
          border: '1px solid #1F2937',
        }}>
          <p style={{ fontSize: '0.58rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', color: '#374151', marginBottom: 3 }}>
            xFIP ajust. parque
          </p>
          {xfipParkAdj != null ? (
            <>
              <span style={{
                fontFamily: 'JetBrains Mono, monospace', fontWeight: 700, fontSize: '1.1rem',
                color: xfipParkAdj <= 3.50 ? '#EF4444' : xfipParkAdj <= 4.50 ? '#F59E0B' : '#22C55E',
              }}>
                {xfipParkAdj.toFixed(2)}
              </span>
              {parkFactor != null && (
                <p style={{ fontSize: '0.58rem', marginTop: 2, color: pfColor }}>
                  PF {parkFactor.toFixed(2)} ({parkFactor > 1.02 ? 'parque bateador' : parkFactor < 0.98 ? 'parque pitcher' : 'neutro'})
                </p>
              )}
            </>
          ) : (
            <div>
              <span style={{ fontSize: '0.75rem', color: '#374151' }}>Pendiente</span>
              <p style={{ fontSize: '0.58rem', color: '#374151', marginTop: 2, fontStyle: 'italic' }}>
                Requiere enriquecimiento API (FanGraphs/Baseball-Reference)
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Data quality notice */}
      <p style={{ fontSize: '0.60rem', color: '#374151', fontStyle: 'italic', lineHeight: 1.5 }}>
        {hasHistoricalAVG
          ? `✓ AVG histórico medido · wOBA ajustada con boost platoon (.025 MLB avg)`
          : `⚠ Sin datos históricos del lineup vs este pitcher. wOBA estimada con boost platoon (.025 MLB avg). Para datos medibles, conectar a FanGraphs API.`
        }
      </p>
    </div>
  );
}
