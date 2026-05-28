/**
 * GameContext — strict game selection state management (Bug 1 fix).
 *
 * Design invariant: when game_pk or team changes, ALL derived data
 * (lineupData) is cleared to null SYNCHRONOUSLY in the reducer BEFORE
 * any fetch begins. This prevents stale data from being rendered while
 * a new request is in flight.
 *
 * Race-condition guard: every fetch is tagged with a monotonic requestId.
 * A response whose requestId doesn't match the current one is discarded,
 * preventing out-of-order updates from fast tab-switching.
 */
import React, { createContext, useContext, useReducer, useEffect } from 'react';
import type { LineupData } from '../types';
import { fetchLineup } from '../api/client';

// ── State ────────────────────────────────────────────────────────────────────

export interface WarRoomState {
  game_pk: number | null;
  team: 'home' | 'away';
  lineupData: LineupData | null;
  lineupLoading: boolean;
  lineupError: string | null;
  /** Monotonic counter; increments on every game_pk or team change. */
  _reqId: number;
}

const initialState: WarRoomState = {
  game_pk: null,
  team: 'home',
  lineupData: null,
  lineupLoading: false,
  lineupError: null,
  _reqId: 0,
};

// ── Actions ───────────────────────────────────────────────────────────────────

export type GameAction =
  | { type: 'GAME_CHANGED'; payload: { game_pk: number } }
  | { type: 'TEAM_CHANGED'; payload: { team: 'home' | 'away' } }
  | { type: 'LINEUP_SUCCESS'; payload: { data: LineupData; reqId: number } }
  | { type: 'LINEUP_ERROR'; payload: { error: string; reqId: number } };

// ── Reducer ───────────────────────────────────────────────────────────────────

function gameReducer(state: WarRoomState, action: GameAction): WarRoomState {
  switch (action.type) {
    case 'GAME_CHANGED':
      if (state.game_pk === action.payload.game_pk) return state;
      return {
        ...state,
        game_pk: action.payload.game_pk,
        lineupData: null,    // ← sync clear — no stale data visible
        lineupLoading: true,
        lineupError: null,
        _reqId: state._reqId + 1,
      };

    case 'TEAM_CHANGED':
      if (state.team === action.payload.team) return state;
      return {
        ...state,
        team: action.payload.team,
        lineupData: null,
        lineupLoading: true,
        lineupError: null,
        _reqId: state._reqId + 1,
      };

    case 'LINEUP_SUCCESS':
      // Discard if a newer request has already superseded this one
      if (action.payload.reqId !== state._reqId) return state;
      return {
        ...state,
        lineupData: action.payload.data,
        lineupLoading: false,
        lineupError: null,
      };

    case 'LINEUP_ERROR':
      if (action.payload.reqId !== state._reqId) return state;
      return {
        ...state,
        lineupData: null,
        lineupLoading: false,
        lineupError: action.payload.error,
      };

    default:
      return state;
  }
}

// ── Context ───────────────────────────────────────────────────────────────────

interface GameContextValue {
  state: WarRoomState;
  dispatch: React.Dispatch<GameAction>;
}

const GameContext = createContext<GameContextValue | null>(null);

// ── Provider ──────────────────────────────────────────────────────────────────

export function GameProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(gameReducer, initialState);

  /**
   * Fetch lineup whenever the selection changes.
   * Captures game_pk, team, and reqId at the moment the effect fires —
   * any concurrent renders can't change these captured values.
   */
  useEffect(() => {
    if (state.game_pk === null) return;

    const { game_pk, team, _reqId: reqId } = state;
    let cancelled = false;

    (async () => {
      try {
        const data = await fetchLineup(game_pk, team);
        if (cancelled) return;

        // Validate API response coherence (Bug 1 — data consistency check)
        if (data.game_pk !== game_pk) {
          throw new Error(
            `API response mismatch: expected game_pk=${game_pk}, got ${data.game_pk}`
          );
        }

        dispatch({ type: 'LINEUP_SUCCESS', payload: { data, reqId } });
      } catch (e) {
        if (!cancelled) {
          dispatch({
            type: 'LINEUP_ERROR',
            payload: {
              error: e instanceof Error ? e.message : 'Error cargando lineup',
              reqId,
            },
          });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
    // Effect fires when game_pk, team, or _reqId changes.
    // _reqId is the canonical trigger: it increments only when game_pk or team changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.game_pk, state.team, state._reqId]);

  return (
    <GameContext.Provider value={{ state, dispatch }}>
      {children}
    </GameContext.Provider>
  );
}

// ── Consumer hook ─────────────────────────────────────────────────────────────

export function useGameContext(): GameContextValue {
  const ctx = useContext(GameContext);
  if (!ctx) {
    throw new Error('useGameContext must be used within <GameProvider>');
  }
  return ctx;
}
