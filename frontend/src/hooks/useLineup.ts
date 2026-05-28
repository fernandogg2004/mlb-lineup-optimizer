import { useState, useEffect, useCallback } from 'react';
import type { LineupData } from '../types';
import { fetchLineup } from '../api/client';

interface UseLineupResult {
  data: LineupData | null;
  loading: boolean;
  error: string | null;
  refetch: () => void;
}

export function useLineup(
  game_pk: number | null,
  team: 'home' | 'away'
): UseLineupResult {
  const [data, setData] = useState<LineupData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  const refetch = useCallback(() => {
    setTick((t) => t + 1);
  }, []);

  useEffect(() => {
    if (game_pk === null) return;

    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const result = await fetchLineup(game_pk!, team, 10000);
        if (!cancelled) {
          setData(result);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err instanceof Error ? err.message : 'Error fetching lineup'
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
    };
  }, [game_pk, team, tick]);

  return { data, loading, error, refetch };
}
