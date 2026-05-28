import { useEffect, useRef } from 'react';
import { clsx } from 'clsx';
import type { WhatIfResult } from '../../types';

interface WhatIfPanelProps {
  result: WhatIfResult | null;
  onReset: () => void;
  onConfirm?: () => void;
}

export default function WhatIfPanel({ result, onReset, onConfirm }: WhatIfPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null);

  // Flash animation on new result
  useEffect(() => {
    if (result && panelRef.current) {
      panelRef.current.classList.remove('animate-flash');
      void panelRef.current.offsetWidth; // reflow
      panelRef.current.classList.add('animate-flash');
    }
  }, [result]);

  if (!result) {
    return (
      <div
        className="flex items-center gap-2 h-12 px-3 rounded-lg border border-dashed"
        style={{ borderColor: '#1F2937' }}
      >
        <span className="text-slate-600 text-base">›</span>
        <p className="text-sm text-slate-500">
          El lineup actual está optimizado. Sin swaps recomendados.
        </p>
      </div>
    );
  }

  const { delta_er, delta_win_probability, base_expected_runs, new_expected_runs,
          base_win_probability, new_win_probability } = result;
  const deltaWP = delta_win_probability;

  const sign = (v: number) => (v >= 0 ? '+' : '');
  const erColor = delta_er >= 0 ? '#00FF87' : '#FF4560';
  const wpColor = deltaWP >= 0 ? '#00FF87' : '#FF4560';
  // Threshold: changes <0.03 E[R] are within noise (audit: show if meaningful)
  const isMeaningful = Math.abs(delta_er) >= 0.03;

  return (
    <div
      ref={panelRef}
      className="rounded-lg border p-4"
      style={{ borderColor: '#1F2937', background: '#0D1117' }}
    >
      {/* Absolute values row — always visible (audit: show absolute, not just delta) */}
      <div className="grid grid-cols-2 gap-3 mb-3">
        <div style={{ background: '#111827', borderRadius: 8, padding: '8px 12px', border: '1px solid #1F2937' }}>
          <p className="text-[9px] text-slate-600 uppercase tracking-widest mb-1">E[R] Baseline</p>
          <p className="text-xl font-mono font-bold text-slate-300">{base_expected_runs.toFixed(2)}</p>
        </div>
        <div style={{ background: '#111827', borderRadius: 8, padding: '8px 12px', border: `1px solid ${erColor}30` }}>
          <p className="text-[9px] text-slate-600 uppercase tracking-widest mb-1">E[R] What-If</p>
          <p className="text-xl font-mono font-bold" style={{ color: erColor }}>{new_expected_runs.toFixed(2)}</p>
        </div>
      </div>

      {/* Delta metrics row */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <MetricCell
          label="ΔE[R]"
          value={`${sign(delta_er)}${delta_er.toFixed(3)}`}
          color={isMeaningful ? erColor : '#6B7280'}
          sub={isMeaningful ? undefined : 'ruido estadístico'}
        />
        <MetricCell
          label="ΔP(W)"
          value={`${sign(deltaWP)}${(deltaWP * 100).toFixed(1)}pp`}
          color={Math.abs(deltaWP * 100) >= 1.0 ? wpColor : '#6B7280'}
        />
        <MetricCell
          label="P(W) nueva"
          value={`${(new_win_probability * 100).toFixed(1)}%`}
          color={wpColor}
        />
      </div>

      {/* Noise notice */}
      {!isMeaningful && (
        <p className="text-xs text-center text-slate-600 mb-3">
          ΔE[R] &lt; 0.03 — dentro del ruido estadístico del modelo
        </p>
      )}

      {/* Baseline win probability context */}
      <p className="text-xs font-mono text-slate-500 mb-3 text-center">
        P(W) actual:{' '}
        <span className="text-slate-300 font-bold">{(base_win_probability * 100).toFixed(1)}%</span>
        {' → '}
        <span style={{ color: wpColor }} className="font-bold">
          {(new_win_probability * 100).toFixed(1)}%
        </span>
      </p>

      {/* Buttons */}
      <div className="flex gap-3">
        <button
          onClick={onReset}
          className={clsx(
            'flex-1 py-2 rounded-lg text-sm font-semibold border transition-colors',
            'text-slate-300 border-slate-600 hover:bg-slate-800'
          )}
        >
          ↺ Revertir
        </button>
        <button
          onClick={onConfirm}
          className="flex-1 py-2 rounded-lg text-sm font-semibold transition-colors"
          style={{
            background: '#00FF87',
            color: '#0A0E1A',
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = '#00E87A';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = '#00FF87';
          }}
        >
          ✓ Confirmar
        </button>
      </div>
    </div>
  );
}

function MetricCell({
  label,
  value,
  color,
  sub,
}: {
  label: string;
  value: string;
  color: string;
  sub?: string;
}) {
  return (
    <div className="flex flex-col items-center gap-1">
      <p className="text-xs text-slate-500 uppercase tracking-widest">{label}</p>
      <p
        className="text-xl font-mono font-bold leading-none"
        style={{ color }}
      >
        {value}
      </p>
      {sub && <p className="text-[9px] text-slate-600 text-center">{sub}</p>}
    </div>
  );
}
