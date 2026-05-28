import type { WinProbFactor } from '../../types';

interface WinProbXAIProps {
  factors: WinProbFactor[];
  basePct: number;
  finalPct: number;
}

/** Waterfall attribution chart — Bloque 8 (XAI). */
export default function WinProbXAI({ factors, basePct, finalPct }: WinProbXAIProps) {
  const sorted  = [...factors].sort((a, b) => Math.abs(b.delta_pct) - Math.abs(a.delta_pct));
  const maxAbs  = Math.max(...sorted.map((f) => Math.abs(f.delta_pct)), 1);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between mb-1">
        <p className="text-xs uppercase tracking-widest text-slate-500">
          ¿Por qué {finalPct.toFixed(1)}%? — Atribución de Factores
        </p>
        <span className="text-[10px] font-mono text-slate-600">
          Base: {basePct.toFixed(1)}%
        </span>
      </div>

      {sorted.map((f, i) => {
        const barW   = (Math.abs(f.delta_pct) / maxAbs) * 100;
        const color  = f.direction === 'positive' ? '#00FF87' : '#FF4560';
        const isPos  = f.direction === 'positive';
        return (
          <div key={i} className="flex items-center gap-2">
            <span
              className="text-[10px] font-mono text-slate-400 shrink-0 w-44 truncate"
              title={f.factor}
            >
              {f.factor}
            </span>
            <div className="flex-1 h-3 rounded bg-slate-800 relative overflow-hidden">
              {/* Positive bars grow from left; negative from right */}
              <div
                className="absolute top-0 h-full rounded"
                style={{
                  width: `${barW}%`,
                  background: color,
                  opacity: 0.75,
                  left: isPos ? 0 : undefined,
                  right: isPos ? undefined : 0,
                }}
              />
            </div>
            <span
              className="text-[10px] font-mono font-bold w-12 text-right shrink-0"
              style={{ color }}
            >
              {f.delta_pct >= 0 ? '+' : ''}{f.delta_pct.toFixed(1)}pp
            </span>
          </div>
        );
      })}
    </div>
  );
}
