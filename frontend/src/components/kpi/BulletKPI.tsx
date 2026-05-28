import { useState } from 'react';
import { clsx } from 'clsx';

interface BulletKPIProps {
  label: string;
  value: number;
  unit: string;
  targetValue: number;
  targetLabel: string;
  referenceValue?: number;
  referenceLabel?: string;
  referenceTooltip?: string;
}

export default function BulletKPI({
  label,
  value,
  unit,
  targetValue,
  targetLabel,
  referenceValue,
  referenceLabel,
  referenceTooltip,
}: BulletKPIProps) {
  const [tooltipOpen, setTooltipOpen] = useState(false);
  const valueColor =
    value > targetValue * 1.1
      ? '#00FF87'
      : value > targetValue * 0.7
      ? '#FFB800'
      : '#FF4560';

  // Clamp percentages for bar display
  const barPct = Math.min(100, Math.max(0, value));
  const targetPct = Math.min(100, Math.max(0, targetValue));
  const refPct =
    referenceValue !== undefined
      ? Math.min(100, Math.max(0, referenceValue))
      : undefined;

  return (
    <div className="flex flex-col gap-1.5">
      {/* Label */}
      <p className="text-xs uppercase tracking-widest text-slate-500">{label}</p>

      {/* Value */}
      <p
        className="text-3xl font-mono font-bold leading-none"
        style={{ color: valueColor }}
      >
        {value.toFixed(1)}
        <span className="text-lg ml-0.5 font-normal">{unit}</span>
      </p>

      {/* Bullet bar track */}
      <div className="relative w-full h-3 rounded-full bg-slate-800">
        {/* Filled bar */}
        <div
          className="absolute left-0 top-0 h-full rounded-full transition-all duration-500"
          style={{
            width: `${barPct}%`,
            background: valueColor,
            opacity: 0.85,
          }}
        />

        {/* Reference line */}
        {refPct !== undefined && (
          <div
            className="absolute top-0 h-full w-0.5 bg-white opacity-60"
            style={{ left: `${refPct}%` }}
            title={referenceLabel}
          />
        )}

        {/* Target line */}
        <div
          className="absolute top-0 h-full w-0.5 bg-slate-400"
          style={{ left: `${targetPct}%` }}
          title={targetLabel}
        />
      </div>

      {/* Sub-text */}
      <p className={clsx('text-xs text-slate-500')}>
        Obj: {targetValue.toFixed(1)}{unit}
        {referenceLabel && refPct !== undefined && (
          <span className="ml-2 text-slate-600 relative">
            ·{' '}
            {referenceTooltip ? (
              <button
                className="underline decoration-dotted cursor-help hover:text-slate-400 transition-colors"
                onMouseEnter={() => setTooltipOpen(true)}
                onMouseLeave={() => setTooltipOpen(false)}
                onClick={() => setTooltipOpen((o) => !o)}
              >
                {referenceLabel}
              </button>
            ) : (
              referenceLabel
            )}
            {tooltipOpen && referenceTooltip && (
              <span
                className="absolute z-30 left-0 bottom-full mb-1.5 rounded-lg border px-3 py-2 text-xs font-mono w-64 shadow-xl whitespace-normal"
                style={{
                  background: '#0D1117',
                  borderColor: '#374151',
                  color: '#9CA3AF',
                }}
              >
                {referenceTooltip}
              </span>
            )}
          </span>
        )}
      </p>
    </div>
  );
}
