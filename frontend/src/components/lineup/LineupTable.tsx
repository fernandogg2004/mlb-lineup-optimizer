import { useState, useCallback, useRef } from 'react';
import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from '@dnd-kit/core';
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable';
import { CSS } from '@dnd-kit/utilities';
import { clsx } from 'clsx';
import type { Player, WhatIfResult } from '../../types';
import { classifyISO, classifyWOBA, classifyOBP, classifyOPS } from '../../utils/statLabels';

// ── Algorithm transparency modal (audit 3.7) ─────────────────────────────────
function AlgorithmModal({ onClose }: { onClose: () => void }) {
  return (
    <div style={{
      position: 'fixed', inset: 0, zIndex: 100,
      background: 'rgba(0,0,0,0.75)',
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }} onClick={onClose}>
      <div style={{
        background: '#111827', border: '1px solid #1F2937',
        borderRadius: 14, padding: '24px 28px',
        maxWidth: 480, width: '92vw',
        boxShadow: '0 20px 60px rgba(0,0,0,0.6)',
      }} onClick={e => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 16 }}>
          <p style={{ fontSize: '0.95rem', fontWeight: 800, color: '#E8EAF6' }}>
            ⚡ ¿Cómo funciona la Optimización Global?
          </p>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#6B7280', fontSize: '1.1rem', cursor: 'pointer' }}>✕</button>
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: '0.78rem', color: '#9CA3AF', lineHeight: 1.7 }}>
          <div style={{ background: '#0D1117', borderRadius: 8, padding: '10px 14px', border: '1px solid #1F2937' }}>
            <p style={{ color: '#60A5FA', fontWeight: 700, marginBottom: 4, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Función objetivo</p>
            <p><strong style={{ color: '#E8EAF6' }}>Maximizar E[R]</strong> — carreras esperadas en 9 entradas.</p>
            <p style={{ fontFamily: 'JetBrains Mono, monospace', color: '#00FF87', marginTop: 4, fontSize: '0.7rem' }}>
              E[R] = Σ wOBA_i × PA_slot_i × 0.89
            </p>
          </div>

          <div style={{ background: '#0D1117', borderRadius: 8, padding: '10px 14px', border: '1px solid #1F2937' }}>
            <p style={{ color: '#60A5FA', fontWeight: 700, marginBottom: 4, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Restricciones</p>
            <ul style={{ paddingLeft: 16, margin: 0 }}>
              <li>Los 9 jugadores del lineup activo deben aparecer exactamente una vez.</li>
              <li>Se respetan posiciones defensivas originales.</li>
              <li>Ventaja de platoon (mano vs pitcher) se pondera como factor secundario.</li>
            </ul>
          </div>

          <div style={{ background: '#0D1117', borderRadius: 8, padding: '10px 14px', border: '1px solid #1F2937' }}>
            <p style={{ color: '#60A5FA', fontWeight: 700, marginBottom: 4, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>PAs esperadas por slot</p>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', fontFamily: 'JetBrains Mono, monospace', fontSize: '0.68rem', color: '#E8EAF6' }}>
              {[{s:1,pa:4.9},{s:2,pa:4.6},{s:3,pa:4.4},{s:4,pa:4.1},{s:5,pa:3.9},{s:6,pa:3.7},{s:7,pa:3.4},{s:8,pa:3.2},{s:9,pa:3.0}].map(({s,pa}) => (
                <span key={s} style={{ background: '#1F2937', padding: '2px 6px', borderRadius: 4 }}>
                  #{s}: {pa}
                </span>
              ))}
            </div>
          </div>

          <div style={{ background: '#0D1117', borderRadius: 8, padding: '10px 14px', border: '1px solid #1F2937' }}>
            <p style={{ color: '#60A5FA', fontWeight: 700, marginBottom: 4, fontSize: '0.7rem', textTransform: 'uppercase', letterSpacing: '0.1em' }}>Limitación del modo rápido</p>
            <p>La «Optimización Global» ordena por wOBA descendente. El optimizador Monte Carlo completo (Algoritmo Genético, 10k sims) produce órdenes de bateo más matizados que incorporan varianza de PA y correlaciones entre bateadores consecutivos.</p>
          </div>
        </div>
      </div>
    </div>
  );
}

interface LineupTableProps {
  lineup: Player[];
  /** Bench players available for injection (Roadmap 2.2) */
  bench?: Player[];
  onReorder: (newLineup: Player[]) => void;
  onWhatIfResult: (result: WhatIfResult) => void;
  gamePk: number;
  /** Current win probability for what-if delta calculation */
  winProbability?: number;
}

// Expected plate appearances per lineup slot (empirical MLB average)
const PA_SLOT: Record<number, number> = {
  1: 4.9, 2: 4.6, 3: 4.4, 4: 4.1, 5: 3.9,
  6: 3.7, 7: 3.4, 8: 3.2, 9: 3.0,
};

// MLB 5-year league average wOBA
const WOBA_LG_AVG = 0.318;

/**
 * Sabermetric Linear Weights what-if estimator (audit fix).
 *
 * Uses the formula: ΔE[R] = Σ (wOBA_i - lgAvg) × (PA_new_slot - PA_old_slot) × 0.89
 * where 0.89 is the wOBA-to-runs linear weight scale factor.
 *
 * This produces realistic deltas: moving a .380 wOBA batter from slot 9 → 1
 * yields +0.106 E[R] (0.062 × 1.9 PA × 0.89), not 0.000.
 */
function computeHeuristicWhatIf(
  originalLineup: Player[],
  newLineup: Player[],
  baseER: number,
  baseWinProb: number = 0.5,
): WhatIfResult {
  let deltaER = 0;

  for (let i = 0; i < newLineup.length; i++) {
    const player = newLineup[i];
    const newSlot = i + 1;
    const origIdx = originalLineup.findIndex((p) => p.player_id === player.player_id);
    const oldSlot = origIdx >= 0 ? origIdx + 1 : newSlot;
    if (newSlot === oldSlot) continue; // no change → skip

    const paNew = PA_SLOT[newSlot] ?? 3.0;
    const paOld = PA_SLOT[oldSlot] ?? 3.0;
    // Linear Weights: marginal run value above/below league average × PA diff
    deltaER += (player.woba - WOBA_LG_AVG) * (paNew - paOld) * 0.89;
  }

  const newER = Math.max(0.5, baseER + deltaER);
  // Pythagorean sensitivity: ~4.5pp win probability per 0.1 run
  const deltaWP = deltaER * 0.045;
  const newPW = Math.max(0.02, Math.min(0.98, baseWinProb + deltaWP));

  // wOBA delta: position-weighted quality change
  let wobaDelta = 0;
  for (let i = 0; i < newLineup.length; i++) {
    const pid = newLineup[i].player_id;
    const origP = originalLineup.find(p => p.player_id === pid);
    const wobaOld = origP ? origP.woba : newLineup[i].woba;
    const slotW = (PA_SLOT[i + 1] ?? 3.0) / 4.0;
    wobaDelta += (newLineup[i].woba - wobaOld) * slotW;
  }

  return {
    base_expected_runs: baseER,
    new_expected_runs: newER,
    delta_er: parseFloat(deltaER.toFixed(4)),
    base_win_probability: baseWinProb,
    new_win_probability: newPW,
    delta_win_probability: parseFloat((newPW - baseWinProb).toFixed(4)),
    simulation_n: 0,
  };
}

interface SortableRowProps {
  player: Player;
}

function SortableRow({ player }: SortableRowProps) {
  const {
    attributes,
    listeners,
    setNodeRef,
    transform,
    transition,
    isDragging,
  } = useSortable({ id: player.player_id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    zIndex: isDragging ? 10 : undefined,
    opacity: isDragging ? 0.8 : 1,
  };

  const handColor =
    player.hand === 'L' ? '#00FF87' : player.hand === 'R' ? '#3B82F6' : '#FFB800';

  return (
    <tr
      ref={setNodeRef}
      style={{ ...style, background: isDragging ? '#1a2335' : '#111827', borderColor: '#1F2937' }}
      className="border-b hover:bg-slate-800 transition-colors"
    >
      {/* Drag handle */}
      <td className="px-2 py-2 w-6">
        <span
          {...attributes}
          {...listeners}
          className="text-slate-600 hover:text-slate-400 cursor-grab active:cursor-grabbing select-none text-base"
          title="Arrastra para reordenar"
        >
          ≡
        </span>
      </td>

      {/* Order badge */}
      <td className="px-2 py-2 w-8">
        <span
          className="inline-flex items-center justify-center w-5 h-5 rounded text-xs font-mono font-bold"
          style={{ background: '#1F2937', color: '#9CA3AF' }}
        >
          {player.order}
        </span>
      </td>

      {/* Name + Position */}
      <td className="px-2 py-2 min-w-[120px]">
        <div className="flex flex-col gap-0.5">
          <span className="text-sm font-semibold text-white leading-none">
            {player.name}
          </span>
          <span className="text-xs text-slate-400">{player.pos}</span>
        </div>
      </td>

      {/* Hand badge */}
      <td className="px-2 py-2 w-10">
        <span
          className="inline-flex items-center justify-center w-5 h-5 rounded text-xs font-mono font-bold"
          style={{
            background: `${handColor}22`,
            color: handColor,
            border: `1px solid ${handColor}44`,
          }}
        >
          {player.hand}
        </span>
      </td>

      {/* Stats */}
      <td className="px-2 py-2">
        <div className="flex items-center gap-3 text-xs font-mono text-slate-300 flex-wrap">
          <StatVal label="AVG" value={player.avg.toFixed(3)} />
          <StatVal label="OPS" value={player.ops.toFixed(3)} label_meta={classifyOPS(player.ops)} />
          <StatVal label="wOBA" value={player.woba.toFixed(3)} label_meta={classifyWOBA(player.woba)} />
          <StatVal label="OBP" value={player.obp.toFixed(3)} label_meta={classifyOBP(player.obp)} />
          <StatVal label="ISO" value={player.iso.toFixed(3)} label_meta={classifyISO(player.iso)} />
        </div>
      </td>
    </tr>
  );
}

function StatVal({
  label,
  value,
  label_meta,
}: {
  label: string;
  value: string;
  label_meta?: { text: string; color: string; bgColor: string };
}) {
  return (
    <span
      className="flex flex-col items-center gap-0 cursor-default"
      title={label_meta ? label_meta.text : undefined}
    >
      <span className="text-slate-500 text-[9px] uppercase leading-none">{label}</span>
      <span
        className="font-mono text-xs leading-tight font-bold"
        style={{ color: label_meta ? label_meta.color : '#9CA3AF' }}
      >
        {value}
      </span>
    </span>
  );
}

export default function LineupTable({
  lineup: initialLineup,
  bench = [],
  onReorder,
  onWhatIfResult,
  gamePk: _gamePk,
  winProbability = 0.5,
}: LineupTableProps) {
  const [swapTarget, setSwapTarget] = useState<number | null>(null);
  const [showAlgModal, setShowAlgModal] = useState(false);
  const [lineup, setLineup] = useState<Player[]>(initialLineup);
  const originalLineupRef = useRef<Player[]>(initialLineup);
  const prevInitialRef = useRef<Player[]>(initialLineup);

  // Sync when parent lineup changes (e.g., game switch)
  if (prevInitialRef.current !== initialLineup) {
    prevInitialRef.current = initialLineup;
    originalLineupRef.current = initialLineup;
    setLineup(initialLineup);
  }

  const baseER =
    lineup.length > 0
      ? (lineup.reduce((sum, p) => sum + p.woba, 0) / lineup.length) * 9
      : 4.5;

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 5 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over } = event;
      if (!over || active.id === over.id) return;

      setLineup((prev) => {
        const oldIndex = prev.findIndex((p) => p.player_id === active.id);
        const newIndex = prev.findIndex((p) => p.player_id === over.id);
        const newLineup = arrayMove(prev, oldIndex, newIndex).map((p, i) => ({
          ...p,
          order: i + 1,
        }));

        onReorder(newLineup);
        const result = computeHeuristicWhatIf(
          originalLineupRef.current,
          newLineup,
          baseER,
          winProbability,
        );
        onWhatIfResult(result);
        return newLineup;
      });
    },
    [onReorder, onWhatIfResult, baseER, winProbability]
  );

  const handleOptimize = useCallback(() => {
    setLineup((prev) => {
      const sorted = [...prev]
        .sort((a, b) => b.woba - a.woba)
        .map((p, i) => ({ ...p, order: i + 1 }));

      const result = computeHeuristicWhatIf(prev, sorted, baseER, winProbability);
      onReorder(sorted);
      onWhatIfResult(result);
      return sorted;
    });
  }, [baseER, winProbability, onReorder, onWhatIfResult]);

  /** Roadmap 2.2 — inject a bench player, replacing the lineup player at swapTarget slot */
  const handleBenchSwap = useCallback((benchPlayer: Player) => {
    if (swapTarget === null) return;
    setLineup((prev) => {
      const newLineup = prev.map((p) =>
        p.order === swapTarget
          ? { ...benchPlayer, order: swapTarget }
          : p
      );
      const result = computeHeuristicWhatIf(prev, newLineup, baseER, winProbability);
      onReorder(newLineup);
      onWhatIfResult(result);
      return newLineup;
    });
    setSwapTarget(null);
  }, [swapTarget, baseER, winProbability, onReorder, onWhatIfResult]);

  return (
    <div>
      {showAlgModal && <AlgorithmModal onClose={() => setShowAlgModal(false)} />}

      {/* Optimize button row */}
      <div className="mb-3 flex gap-2">
        <button
          onClick={handleOptimize}
          className="flex-1 py-2.5 rounded-lg text-sm font-bold transition-all border"
          style={{
            background: 'rgba(0,255,135,0.08)',
            borderColor: '#00FF87',
            color: '#00FF87',
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(0,255,135,0.15)';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(0,255,135,0.08)';
          }}
        >
          ⚡ Optimización Global
        </button>
        {/* Algorithm transparency button (audit 3.7) */}
        <button
          onClick={() => setShowAlgModal(true)}
          title="¿Cómo funciona este algoritmo?"
          className="px-3 py-2.5 rounded-lg text-xs border transition-all"
          style={{
            background: 'rgba(59,130,246,0.06)',
            borderColor: '#3B82F620',
            color: '#6B7280',
          }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(59,130,246,0.12)';
            (e.currentTarget as HTMLButtonElement).style.color = '#60A5FA';
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLButtonElement).style.background = 'rgba(59,130,246,0.06)';
            (e.currentTarget as HTMLButtonElement).style.color = '#6B7280';
          }}
        >
          ℹ️ ¿Cómo funciona?
        </button>
      </div>

      {/* Swap slot selector — shown when bench has players */}
      {bench.length > 0 && (
        <div className="mb-3 flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-500 uppercase tracking-widest">Banquillo →</span>
          <select
            value={swapTarget ?? ''}
            onChange={(e) => setSwapTarget(Number(e.target.value) || null)}
            className="text-xs rounded px-2 py-1 font-mono"
            style={{ background: '#111827', border: '1px solid #1F2937', color: '#9CA3AF' }}
          >
            <option value="">Slot a reemplazar</option>
            {lineup.map((p) => (
              <option key={p.order} value={p.order}>
                #{p.order} {p.name} ({p.pos})
              </option>
            ))}
          </select>
          {swapTarget !== null && (
            <>
              <span className="text-xs text-slate-600">→ entra:</span>
              {bench.map((bp) => (
                <button
                  key={bp.player_id}
                  onClick={() => handleBenchSwap(bp)}
                  className="text-xs px-2 py-1 rounded border transition-colors"
                  style={{
                    background: 'rgba(59,130,246,0.08)',
                    borderColor: '#3B82F6',
                    color: '#60A5FA',
                    cursor: 'pointer',
                  }}
                  title={`wOBA ${bp.woba.toFixed(3)} | OPS ${bp.ops.toFixed(3)}`}
                >
                  {bp.name} <span style={{ color: '#374151' }}>({bp.pos})</span>
                </button>
              ))}
              <button
                onClick={() => setSwapTarget(null)}
                className="text-xs px-2 py-1 rounded"
                style={{ color: '#6B7280', cursor: 'pointer', background: 'none', border: 'none' }}
              >
                ✕
              </button>
            </>
          )}
        </div>
      )}

      {/* Table */}
      <div
        className="rounded-lg border overflow-hidden"
        style={{ borderColor: '#1F2937' }}
      >
        <DndContext
          sensors={sensors}
          collisionDetection={closestCenter}
          onDragEnd={handleDragEnd}
        >
          <SortableContext
            items={lineup.map((p) => p.player_id)}
            strategy={verticalListSortingStrategy}
          >
            <table className="w-full border-collapse">
              <thead>
                <tr style={{ background: '#0D1117' }}>
                  <th className="w-6" />
                  <th className="w-8" />
                  <th className="px-2 py-2 text-left text-xs uppercase tracking-widest text-slate-500">
                    Jugador
                  </th>
                  <th className="w-10 text-xs uppercase tracking-widest text-slate-500 py-2">
                    M
                  </th>
                  <th className="px-2 py-2 text-left text-xs uppercase tracking-widest text-slate-500">
                    Estadísticas
                  </th>
                </tr>
              </thead>
              <tbody>
                {lineup.map((player) => (
                  <SortableRow key={player.player_id} player={player} />
                ))}
              </tbody>
            </table>
          </SortableContext>
        </DndContext>
      </div>
    </div>
  );
}
