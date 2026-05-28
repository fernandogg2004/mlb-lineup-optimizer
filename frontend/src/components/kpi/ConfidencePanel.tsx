import { useState } from 'react';
import type { ModelConfidenceDetail } from '../../types';

interface ConfidencePanelProps {
  value: number; // 0–100
  detail?: ModelConfidenceDetail;
}

export default function ConfidencePanel({ value, detail }: ConfidencePanelProps) {
  const [open, setOpen] = useState(false);

  const OPTIMAL  = detail?.optimal_threshold  ?? 80;
  const DEGRADED = detail?.degraded_threshold ?? 70;

  const isDegraded = value < DEGRADED;
  const color      = isDegraded ? '#FF4560' : value >= OPTIMAL ? '#00FF87' : '#FFB800';
  const trend      = value >= OPTIMAL ? '↑' : isDegraded ? '↓' : '→';
  const barPct     = Math.min(100, Math.max(0, value));

  return (
    <div className="flex flex-col gap-1.5 relative">
      {/* Header */}
      <div className="flex items-center justify-between">
        <p className="text-xs uppercase tracking-widest text-slate-500">Confianza</p>
        <span className="font-mono font-bold leading-none" style={{ color, fontSize: '1.5rem' }}>
          {value.toFixed(1)}
          <span className="text-base font-normal ml-0.5">%</span>
          <span className="text-sm ml-1">{trend}</span>
        </span>
      </div>

      {/* Bullet bar */}
      <div className="relative w-full h-3 rounded-full bg-slate-800">
        <div
          className="absolute left-0 top-0 h-full rounded-full transition-all duration-500"
          style={{ width: `${barPct}%`, background: color, opacity: 0.85 }}
        />
        {/* Degraded threshold line */}
        <div
          className="absolute top-0 h-full w-0.5 bg-red-500 opacity-40"
          style={{ left: `${DEGRADED}%` }}
          title={`${DEGRADED}% — umbral mínimo`}
        />
        {/* Optimal threshold line */}
        <div
          className="absolute top-0 h-full w-0.5 bg-slate-400"
          style={{ left: `${Math.min(100, OPTIMAL)}%` }}
          title={`${OPTIMAL}% — óptimo`}
        />
      </div>

      {/* Footer row */}
      <div className="flex items-center justify-between">
        <p className="text-xs text-slate-500">Óptimo: {OPTIMAL}%</p>
        <button
          className="text-[10px] text-slate-600 hover:text-slate-400 transition-colors cursor-help"
          onMouseEnter={() => setOpen(true)}
          onMouseLeave={() => setOpen(false)}
          onClick={() => setOpen((o) => !o)}
        >
          ℹ Metodología
        </button>
      </div>

      {/* Degraded warning */}
      {isDegraded && (
        <div
          className="rounded px-2 py-1 text-xs border"
          style={{
            background: 'rgba(255,69,96,0.08)',
            borderColor: '#FF4560',
            color: '#FF4560',
          }}
        >
          ⚠ Por debajo de {DEGRADED}% — modelo en modo degradado
        </div>
      )}

      {/* Methodology tooltip */}
      {open && (
        <div
          className="absolute z-30 right-0 rounded-lg border p-3 text-xs font-mono w-72 shadow-xl"
          style={{
            top: 'calc(100% + 4px)',
            background: '#0D1117',
            borderColor: '#374151',
            color: '#9CA3AF',
          }}
        >
          <p className="text-white font-bold mb-2 text-[11px]">Desglose de Confianza</p>
          <ul className="space-y-1.5">
            {detail ? (
              <>
                <li>
                  • Backtesting 30d:{' '}
                  {detail.components.backtesting_accuracy_30d !== null && detail.components.backtesting_accuracy_30d !== undefined ? (
                    <>
                      <span className="text-white">
                        {detail.components.backtesting_accuracy_30d.toFixed(0)}% precisión
                      </span>
                      {detail.components.backtesting_sample_n !== null && detail.components.backtesting_sample_n !== undefined && (
                        <span className="text-slate-600">
                          {' '}({detail.components.backtesting_sample_n.toLocaleString()} partidos)
                        </span>
                      )}
                    </>
                  ) : (
                    <span className="text-slate-600">Ejecuta backtest.py para datos reales</span>
                  )}
                </li>
                <li>
                  • Estabilidad MC (10.000 sims): σ ={' '}
                  <span className="text-white">
                    {detail.components.montecarlo_stability_sigma.toFixed(3)}
                  </span>
                </li>
                <li>
                  • Cobertura de datos:{' '}
                  <span className="text-white">
                    {detail.components.data_coverage_pct.toFixed(0)}%
                  </span>{' '}
                  variables disponibles
                </li>
                <li>
                  • Última actualización:{' '}
                  <span className="text-white">hace {detail.components.last_updated_ms}ms</span>
                </li>
              </>
            ) : (
              <>
                <li className="text-slate-500">Desglose detallado no disponible para este partido.</li>
                <li className="text-slate-600 text-[10px] mt-1">
                  El endpoint no devolvió métricas de backtesting.
                </li>
              </>
            )}
          </ul>
          <p className="mt-2 text-[10px] text-slate-600 border-t border-slate-800 pt-2">
            ⚠ Degradado si {'<'}{DEGRADED}% &nbsp;·&nbsp; Óptimo si {'>='}{OPTIMAL}%
          </p>
        </div>
      )}
    </div>
  );
}
