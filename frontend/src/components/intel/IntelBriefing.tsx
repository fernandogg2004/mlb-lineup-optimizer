import { useMemo } from 'react';
import { clsx } from 'clsx';
import type { IntelCard } from '../../types';

interface IntelBriefingProps {
  ragText: string;
  /** Pitcher's throwing hand: 'R', 'L', 'S' (switch), or undefined if unknown. */
  pitcherHand?: string;
  /** Prediction traceability fields (Transversal — trazabilidad) */
  modelVersion?: string;
  totalSimulations?: number;
  elapsedSeconds?: number;
}

function parseIntelCards(text: string): IntelCard[] {
  if (!text) return [];

  const lines = text.split('\n').map((l) => l.trim()).filter(Boolean);

  const hasBullets = lines.some(
    (l) =>
      l.startsWith('✅') ||
      l.startsWith('- ✅') ||
      l.startsWith('🔴') ||
      l.startsWith('- 🔴') ||
      l.startsWith('⚡') ||
      l.startsWith('- ⚡') ||
      l.startsWith('📊') ||
      l.startsWith('- 📊')
  );

  if (!hasBullets) {
    return [{ type: 'advantage', content: text, player: undefined, metric: undefined }];
  }

  const cards: IntelCard[] = [];

  for (const line of lines) {
    const clean = line.replace(/^- /, '');

    let type: IntelCard['type'] | null = null;
    let content = clean;

    if (clean.startsWith('✅')) {
      type = 'advantage';
      content = clean.slice(1).trim();
    } else if (clean.startsWith('🔴')) {
      type = 'risk';
      content = clean.slice(1).trim();
    } else if (clean.startsWith('⚡')) {
      type = 'decision';
      content = clean.slice(1).trim();
    } else if (clean.startsWith('📊')) {
      type = 'matchup';
      content = clean.slice(1).trim();
    }

    if (type) {
      const playerMatch = content.match(/\*\*([^*]+)\*\*/);
      const player = playerMatch ? playerMatch[1] : undefined;
      if (player) content = content.replace(/\*\*[^*]+\*\*/, player);

      const metricMatch = content.match(/(\d+(?:\.\d+)?(?:%|pp)?)/);
      const metric = metricMatch ? metricMatch[1] : undefined;

      cards.push({ type, content, player, metric });
    }
  }

  return cards;
}

const CARD_STYLES: Record<
  IntelCard['type'],
  { bg: string; border: string; icon: string; labelColor: string }
> = {
  advantage: { bg: 'bg-green-950/40',  border: 'border-l-4 border-green-400',  icon: '✅', labelColor: '#00FF87' },
  risk:      { bg: 'bg-red-950/40',    border: 'border-l-4 border-red-500',    icon: '🔴', labelColor: '#FF4560' },
  decision:  { bg: 'bg-amber-950/40',  border: 'border-l-4 border-amber-400',  icon: '⚡', labelColor: '#FFB800' },
  matchup:   { bg: 'bg-blue-950/40',   border: 'border-l-4 border-blue-400',   icon: '📊', labelColor: '#3B82F6' },
};

function CardItem({ card, delay }: { card: IntelCard; delay: number }) {
  const styles = CARD_STYLES[card.type];
  const isRawMarkdown = card.content.length > 200 || card.content.includes('\n');

  return (
    <div
      className={clsx('rounded-r-lg px-3 py-2 animate-card', styles.bg, styles.border)}
      style={{ animationDelay: `${delay}ms` }}
    >
      {isRawMarkdown ? (
        <p className="text-xs text-slate-300 leading-relaxed whitespace-pre-line">
          {card.content}
        </p>
      ) : (
        <div className="flex flex-col gap-0.5">
          {card.player && (
            <span className="text-xs font-bold leading-none" style={{ color: styles.labelColor }}>
              {card.player}
            </span>
          )}
          <p className="text-xs text-slate-300 leading-tight line-clamp-2">
            {card.metric && (
              <span className="font-mono font-bold mr-1" style={{ color: styles.labelColor }}>
                {card.metric}
              </span>
            )}
            {card.content}
          </p>
        </div>
      )}
    </div>
  );
}

/** Pitcher handedness badge shown in the Intel Briefing header (Bloque 1). */
function HandednessBadge({ hand }: { hand: string | undefined }) {
  if (hand === undefined) {
    return (
      <span
        className="text-[10px] font-mono px-2 py-0.5 rounded border"
        style={{ color: '#FFB800', borderColor: '#FFB800', background: 'rgba(255,184,0,0.08)' }}
        title="La lateralidad del pitcher no está disponible. Se muestran métricas generales."
      >
        Handedness no disponible
      </span>
    );
  }
  if (hand === 'S') {
    return (
      <span
        className="text-[10px] font-mono font-bold px-2 py-0.5 rounded border cursor-help"
        style={{ color: '#FFB800', borderColor: '#FFB800', background: 'rgba(255,184,0,0.10)' }}
        title="Switch pitcher: lanza como RHP contra bateadores zurdos y LHP contra diestros."
      >
        SP Switch
      </span>
    );
  }
  const isRight = hand === 'R';
  return (
    <span
      className="text-[10px] font-mono font-bold px-2 py-0.5 rounded border"
      style={{
        color:       isRight ? '#3B82F6' : '#00FF87',
        borderColor: isRight ? '#3B82F6' : '#00FF87',
        background:  isRight ? 'rgba(59,130,246,0.08)' : 'rgba(0,255,135,0.08)',
      }}
    >
      vs {hand}HP
    </span>
  );
}

export default function IntelBriefing({ ragText, pitcherHand, modelVersion, totalSimulations, elapsedSeconds }: IntelBriefingProps) {
  const cards = useMemo(() => parseIntelCards(ragText), [ragText]);

  const advantages = cards.filter((c) => c.type === 'advantage').slice(0, 3);
  const risks      = cards.filter((c) => c.type === 'risk').slice(0, 3);
  const decisions  = cards.filter((c) => c.type === 'decision').slice(0, 3);
  const matchups   = cards.filter((c) => c.type === 'matchup').slice(0, 3);

  const isRawMarkdown =
    cards.length === 1 &&
    cards[0].type === 'advantage' &&
    (cards[0].content.length > 200 || cards[0].content.includes('\n'));

  return (
    <div
      className="rounded-xl border p-5"
      style={{ background: '#111827', borderColor: '#1F2937' }}
    >
      {/* Header with pitcher context badge (Bloque 1) */}
      <div className="flex items-center justify-between mb-4">
        <p className="text-xs uppercase tracking-widest text-slate-500">Intel Briefing</p>
        <HandednessBadge hand={pitcherHand} />
      </div>

      {/* Prediction traceability row (Transversal — trazabilidad) */}
      {(modelVersion || totalSimulations) && (
        <div style={{
          display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap',
          fontSize: '0.62rem', color: '#2E5580',
          fontFamily: 'JetBrains Mono, monospace',
          borderTop: '1px solid #162030', paddingTop: 8, marginTop: -4,
        }}>
          {modelVersion && <span>🔖 modelo {modelVersion}</span>}
          {totalSimulations && <span>🎲 {totalSimulations.toLocaleString()} sims</span>}
          {elapsedSeconds !== undefined && (
            <span>⏱ {elapsedSeconds < 1
              ? `${(elapsedSeconds * 1000).toFixed(0)}ms`
              : `${elapsedSeconds.toFixed(2)}s`}
            </span>
          )}
          <span style={{ marginLeft: 'auto', color: '#1E3850' }}>
            MLB Stats API · {new Date().toLocaleTimeString('es-ES', { hour: '2-digit', minute: '2-digit' })}
          </span>
        </div>
      )}

      {!ragText || cards.length === 0 ? (
        <p className="text-sm text-slate-600 text-center py-4">Sin análisis disponible</p>
      ) : isRawMarkdown ? (
        <div
          className="rounded-r-lg px-4 py-3 animate-card border-l-4"
          style={{ background: 'rgba(59,130,246,0.08)', borderColor: '#3B82F6' }}
        >
          <p className="text-sm text-slate-300 leading-relaxed whitespace-pre-line">{ragText}</p>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-3">
            <div className="flex flex-col gap-2">
              {advantages.length > 0 && (
                <p className="text-[10px] uppercase tracking-widest text-slate-600">Ventajas</p>
              )}
              {advantages.map((card, i) => (
                <CardItem key={i} card={card} delay={i * 150} />
              ))}
            </div>
            <div className="flex flex-col gap-2">
              {risks.length > 0 && (
                <p className="text-[10px] uppercase tracking-widest text-slate-600">Riesgos</p>
              )}
              {risks.map((card, i) => (
                <CardItem key={i} card={card} delay={i * 150 + 100} />
              ))}
            </div>
          </div>

          {(decisions.length > 0 || matchups.length > 0) && (
            <div className="grid grid-cols-2 gap-3">
              <div className="flex flex-col gap-2">
                {decisions.length > 0 && (
                  <p className="text-[10px] uppercase tracking-widest text-slate-600">Decisiones</p>
                )}
                {decisions.map((card, i) => (
                  <CardItem key={i} card={card} delay={i * 150 + 300} />
                ))}
              </div>
              <div className="flex flex-col gap-2">
                {matchups.length > 0 && (
                  <p className="text-[10px] uppercase tracking-widest text-slate-600">Matchups</p>
                )}
                {matchups.map((card, i) => (
                  <CardItem key={i} card={card} delay={i * 150 + 400} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
