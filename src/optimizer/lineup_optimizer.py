"""
lineup_optimizer.py
===================
Three-layer combinatorial lineup optimizer that finds the batting order
permutation maximizing Expected Runs E[R] scored per game.

Search space
------------
9! = 362,880 valid batting orders. Exhaustive evaluation at 5,000 simulations
each would require ~50 CPU-hours — infeasible within the 90-minute SLA.

Three-layer strategy
--------------------
Layer 1 — Sabermetric Seeding:
    Generate a population of 200 individuals using domain-knowledge heuristics
    (OBP-based leaders in slots 1–2, ISO/wOBA in slots 3–5). This concentrates
    the initial search near historically-optimal regions of the fitness landscape.

Layer 2 — Genetic Algorithm (DEAP):
    Run 150 generations × 200 individuals using:
    - Ordered Crossover (OX): preserves relative batter ordering from both parents.
    - Swap Mutation: randomly exchanges two lineup positions.
    - Tournament Selection (k=3): maintains selection pressure without premature convergence.
    - Fitness: E[R] from a 5,000-simulation fast proxy (Numba-only, no Ray overhead).

Layer 3 — High-Fidelity Refinement:
    Evaluate the top-10 GA candidates with the full 100,000-simulation engine
    and select the global winner. This eliminates noise in the fast proxy.

Ordered Crossover (OX) for permutations
----------------------------------------
Given parents P1 = [2,8,4,5,6,7,1,3,9] and P2 = [5,4,6,8,9,2,3,7,1]:
  1. Select cut points (a=3, b=6): preserve segment P1[3:6] = [5,6,7].
  2. Child = [_,_,_,5,6,7,_,_,_]
  3. Fill remaining slots from P2 in order, skipping elements in {5,6,7}:
     P2 filtered = [4,8,9,2,3,1]  →  child = [4,8,9,5,6,7,2,3,1]

OX guarantees:
  - Every batter appears exactly once (valid permutation).
  - The relative ordering of non-copied batters is inherited from P2.
  - No post-crossover repair needed.

Usage:
    from src.optimizer.lineup_optimizer import GeneticLineupOptimizer, PlayerStats
    optimizer = GeneticLineupOptimizer(
        players=[PlayerStats(id=i, ...) for i in range(9)],
        pa_predictor=predictor,
        simulation_engine=engine,
    )
    result = optimizer.run(opp_lineup_probs, park_factor_hr=1.0)
    print(result.best_lineup_ids)
    print(f"E[R] = {result.best_expected_runs:.3f}")
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import mlflow
import numpy as np
import structlog
from deap import base, creator, tools

from src.constants import N_OUTCOMES

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
)
log: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Player statistics contract
# ---------------------------------------------------------------------------

@dataclass
class PlayerStats:
    """Sabermetric profile used for seeding the initial population.

    Attributes:
        player_id: MLB player identifier.
        player_name: Display name.
        obp: On-Base Percentage (season or rolling 30d).
        woba: Weighted On-Base Average.
        iso: Isolated Power = SLG − AVG.
        batter_stand: Batting hand (``"L"``, ``"R"``, or ``"S"``).
        prob_vector: Pre-computed calibrated probability vector vs. today's
            starting pitcher, shape (8,) — 8 PA outcome classes incl. DOUBLE_PLAY
            (Mejora 6). If ``None``, must be provided later via the full feature
            matrix.
    """

    player_id: int
    player_name: str
    obp: float
    woba: float
    iso: float
    batter_stand: str
    prob_vector: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# GA configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """Hyperparameters for the genetic algorithm.

    Attributes:
        population_size: Number of individuals per generation.
        n_generations: Maximum number of generations to run.
        cx_probability: Crossover probability per pair of parents.
        mut_probability: Probability that an individual is selected for mutation.
        mut_indpb: Per-gene probability of being swapped during mutation.
        tournament_size: Number of individuals in each selection tournament.
        elite_size: Number of top individuals copied unchanged to next generation
            (elitism prevents losing the best solution found so far).
        fast_mode_n_sims: Simulations per fitness evaluation in the GA phase.
        refinement_n_sims: Simulations for the top-K final refinement.
        refinement_top_k: Number of top candidates to evaluate in full.
        seed: Random seed for reproducibility.
        mlflow_experiment: MLflow experiment name.
        n_sabermetric_seeds: Individuals seeded with domain-knowledge heuristics.
            The remainder are random valid permutations.
    """

    population_size: int = 200
    n_generations: int = 150
    cx_probability: float = 0.80
    mut_probability: float = 0.30
    mut_indpb: float = 0.20
    tournament_size: int = 3
    elite_size: int = 5
    fast_mode_n_sims: int = 5_000
    refinement_n_sims: int = 100_000
    refinement_top_k: int = 10
    seed: int = 42
    mlflow_experiment: str = "lineup-optimizer"
    n_sabermetric_seeds: int = 50
    # Objetivo del refinamiento final (audit A2). El GA busca siempre sobre E[R]
    # (proxy barato y muy correlacionado), pero el ganador entre los top-K puede
    # elegirse por:
    #   "expected_runs"    → máximo E[R] (comportamiento histórico; por defecto)
    #   "win_probability"  → máxima P(Victoria) contra el oponente fijo, que es el
    #                        objetivo REAL del producto. Para oponente constante
    #                        ambos correlacionan, pero no son idénticos (la
    #                        varianza del lineup importa).
    objective: str = "expected_runs"


# ---------------------------------------------------------------------------
# Sabermetric initialization heuristics
# ---------------------------------------------------------------------------

class SabermetricSeeder:
    """Generates batting order seeds using baseball domain knowledge.

    The canonical lineup construction principles encoded here are derived
    from "The Book" (Tango-Lichtman-Dolphin) and FanGraphs lineup value research:

        Slot 1 (leadoff):  Highest OBP — maximizes PA count, drives run-scoring
                           opportunities for the heart of the order.
        Slot 2 (two-hole): Second-highest OBP + good contact (modern "best hitter"
                           slot per Tango et al. 2017 research).
        Slots 3–5 (core):  Highest wOBA + ISO — RISP opportunities.
        Slots 6–7 (bottom): Moderate wOBA.
        Slots 8–9 (bottom): Lowest wOBA; slot 8 functions as a second leadoff
                            (bats before slot 9 who bats before slot 1 again).

    Platoon adjustment: When the opponent starter is RHP, prefer LHB in the
    top-5 slots (platoon advantage). This is incorporated as a tie-breaker.

    Args:
        players: List of 9 PlayerStats objects representing today's roster.
    """

    _SLOT_PRIORITIES: dict[int, str] = {
        0: "obp",   # leadoff
        1: "obp",   # two-hole
        2: "woba",  # three-hole
        3: "woba",  # cleanup
        4: "woba",  # five-hole
        5: "woba",  # six-hole
        6: "woba",  # seven-hole
        7: "woba",  # eight-hole (weak)
        8: "woba",  # nine-hole
    }

    def __init__(self, players: list[PlayerStats]) -> None:
        """Initializes the seeder.

        Args:
            players: Exactly 9 PlayerStats objects for today's lineup.

        Raises:
            ValueError: If ``len(players) != 9``.
        """
        if len(players) != 9:
            raise ValueError(f"Expected exactly 9 players, got {len(players)}.")
        self.players = players

    def canonical_seed(self, pitcher_hand: str = "R") -> list[int]:
        """Returns the single best-guess batting order as a list of indices.

        Implements a greedy slot-filling algorithm: fill slots 1–2 by OBP,
        then slots 3–9 by wOBA, with LHB preferred in top slots against RHP
        as a tie-breaker.

        Args:
            pitcher_hand: Opposing pitcher's throwing hand (``"L"`` or ``"R"``).

        Returns:
            List of 9 player indices (into ``self.players``) representing
            the canonical batting order (position 0 = leadoff, 8 = ninth).
        """
        def sort_key(idx: int, primary_stat: str) -> tuple[float, float]:
            p = self.players[idx]
            stat = getattr(p, primary_stat, 0.0)
            # Platoon bonus: LHB vs. RHP gets a small tie-breaking boost
            platoon_bonus = 0.005 if (pitcher_hand == "R" and p.batter_stand == "L") else 0.0
            return (-(stat + platoon_bonus), idx)  # negate for descending sort

        remaining = list(range(9))
        order: list[int] = []

        # Slots 1–2: by OBP
        for slot in range(2):
            remaining.sort(key=lambda i: sort_key(i, "obp"))
            order.append(remaining.pop(0))

        # Slots 3–9: by wOBA
        remaining.sort(key=lambda i: sort_key(i, "woba"))
        order.extend(remaining)

        assert len(order) == 9 and len(set(order)) == 9, "Seeder produced invalid permutation."
        return order

    def diverse_seeds(self, n_seeds: int, pitcher_hand: str = "R") -> list[list[int]]:
        """Generates N diverse sabermetric seeds by introducing controlled noise.

        The canonical seed is the first individual. The remaining ``n_seeds - 1``
        are generated by:
        1. Swapping adjacent batters with probability 0.3 (preserves most structure).
        2. Randomly swapping one top-5 batter with a bottom-4 batter (explores wider).

        This creates a population that is both informed by domain knowledge and
        diverse enough to explore non-obvious configurations.

        Args:
            n_seeds: Number of seeded individuals to generate.
            pitcher_hand: Opposing pitcher's throwing hand.

        Returns:
            List of ``n_seeds`` permutation lists.
        """
        seeds: list[list[int]] = []
        base = self.canonical_seed(pitcher_hand)
        seeds.append(list(base))

        for _ in range(n_seeds - 1):
            candidate = list(base)

            # Controlled perturbation 1: adjacent swap with prob 0.3 per pair
            for j in range(8):
                if random.random() < 0.3:
                    candidate[j], candidate[j + 1] = candidate[j + 1], candidate[j]

            # Controlled perturbation 2: occasionally swap a top and bottom batter
            if random.random() < 0.4:
                top_slot    = random.randint(0, 4)
                bottom_slot = random.randint(5, 8)
                candidate[top_slot], candidate[bottom_slot] = (
                    candidate[bottom_slot], candidate[top_slot]
                )

            assert len(set(candidate)) == 9, "Seeder perturbation produced duplicate."
            seeds.append(candidate)

        return seeds


# ---------------------------------------------------------------------------
# DEAP crossover and mutation operators
# ---------------------------------------------------------------------------

def cxOrdered(ind1: list[int], ind2: list[int]) -> tuple[list[int], list[int]]:
    """Ordered Crossover (OX) for permutation individuals.

    OX is the canonical crossover operator for permutation-encoded GAs.
    It guarantees that both offspring are valid permutations (no duplicates,
    no missing elements) without any post-crossover repair step.

    Algorithm (for child1):
        1. Select random cut points a < b.
        2. Copy ind1[a:b] directly to child1[a:b].
        3. Fill remaining positions in child1 from ind2, reading left-to-right
           (wrapping around), skipping elements already in the copied segment.

    Args:
        ind1: First parent permutation, modified in-place.
        ind2: Second parent permutation, modified in-place.

    Returns:
        Tuple (child1, child2) as modified in-place views of ind1 and ind2.
    """
    size = len(ind1)
    a, b = sorted(random.sample(range(size), 2))

    # --- Child 1: segment from ind1, remainder from ind2 ---
    seg1    = ind1[a:b]
    seg1_set = set(seg1)
    fill1   = [x for x in ind2 if x not in seg1_set]
    child1  = fill1[:a] + seg1 + fill1[a:]

    # --- Child 2: segment from ind2, remainder from ind1 ---
    seg2    = ind2[a:b]
    seg2_set = set(seg2)
    fill2   = [x for x in ind1 if x not in seg2_set]
    child2  = fill2[:a] + seg2 + fill2[a:]

    ind1[:] = child1
    ind2[:] = child2
    return ind1, ind2


def mutSwapPositions(individual: list[int], indpb: float) -> tuple[list[int]]:
    """Swap mutation: exchanges two random positions in the lineup.

    This is the simplest valid mutation for permutations: swapping any two
    elements produces another valid permutation. The ``indpb`` parameter
    controls how aggressively mutations are applied — higher values swap
    more positions per application.

    Applies repeated pairwise swaps with probability ``indpb`` per position,
    so on average ``len(individual) × indpb`` positions are swapped per call.

    Args:
        individual: Permutation list, modified in-place.
        indpb: Per-position probability of participating in a swap.

    Returns:
        Tuple containing the modified individual (DEAP convention).
    """
    size = len(individual)
    for i in range(size):
        if random.random() < indpb:
            j = random.randint(0, size - 1)
            if i != j:
                individual[i], individual[j] = individual[j], individual[i]
    return (individual,)


# ---------------------------------------------------------------------------
# Optimization result
# ---------------------------------------------------------------------------

@dataclass
class OptimizerResult:
    """Output of a full optimizer run.

    Attributes:
        best_lineup_indices: Optimal permutation of player indices (0–8).
        best_lineup_ids: Corresponding MLB player IDs.
        best_lineup_names: Corresponding player names for display.
        best_expected_runs: E[R] from the full 100k-simulation refinement.
        best_win_probability: P(W) from the full refinement.
        ga_best_er_history: E[R] of the best individual per generation (fast proxy).
        ga_avg_er_history: Average E[R] per generation (fast proxy).
        refinement_scores: E[R] scores for each of the top-K candidates.
        total_elapsed_seconds: Wall-clock time for the entire optimizer run.
        n_fitness_evaluations: Total number of fitness function calls.
    """

    best_lineup_indices: list[int]
    best_lineup_ids: list[int]
    best_lineup_names: list[str]
    best_expected_runs: float
    best_win_probability: float
    ga_best_er_history: list[float]
    ga_avg_er_history: list[float]
    refinement_scores: list[float]
    total_elapsed_seconds: float
    n_fitness_evaluations: int
    # ── Significancia estadística del orden ganador (audit F18) ───────────────
    best_expected_runs_se: float = 0.0      # error estándar de muestreo del E[R]
    is_significant_vs_second: bool = True   # ¿el #1 supera al #2 más allá del ruido?

    def display(self) -> str:
        """Returns a formatted lineup display string for the coaching staff UI."""
        lines = [
            "=== OPTIMAL LINEUP ===",
            f"E[R] = {self.best_expected_runs:.3f}  P(W) = {self.best_win_probability:.3f}",
            f"Time: {self.total_elapsed_seconds:.1f}s | Evaluations: {self.n_fitness_evaluations:,}",
            "",
        ]
        for slot, (idx, pid, name) in enumerate(zip(
            self.best_lineup_indices, self.best_lineup_ids, self.best_lineup_names
        ), start=1):
            lines.append(f"  {slot}. {name:25s}  (id={pid})")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main optimizer class
# ---------------------------------------------------------------------------

class GeneticLineupOptimizer:
    """Finds the E[R]-maximizing batting order using a 3-layer search strategy.

    The optimizer is stateless between runs: each call to ``run()`` is fully
    independent.

    Reproducibilidad (audit F15): ``config.seed`` siembra los operadores del GA
    (población inicial, crossover, mutación, selección), de modo que la
    TRAYECTORIA de búsqueda es reproducible. Sin embargo, la fitness Monte Carlo
    corre sobre ``numba.prange`` con un estado RNG por hilo no sembrado, así que
    los valores ABSOLUTOS de E[R] tienen un pequeño ruido de muestreo (~1/√n_sims)
    que varía entre ejecuciones y número de hilos. El ranking de alineaciones es
    estable; los E[R] reportados deben interpretarse con su error de muestreo (ver
    ``SimulationResult.win_prob_ci_*`` y el IC de E[R]), no como cifras exactas.

    Args:
        players: List of exactly 9 ``PlayerStats`` for today's active roster.
        lineup_probs: Pre-computed probability matrix, shape (9, 8).
            Row i corresponds to ``players[i]``. Each row must sum to 1.0.
        simulation_engine: Configured ``MonteCarloEngine`` instance.
        config: GA hyperparameters and simulation settings.
    """

    def __init__(
        self,
        players: list[PlayerStats],
        lineup_probs: np.ndarray,
        simulation_engine,           # MonteCarloEngine (typed as Any to avoid circular import)
        opp_lineup_probs: np.ndarray,
        config: Optional[GAConfig] = None,
    ) -> None:
        """Initializes the optimizer.

        Args:
            players: Exactly 9 ``PlayerStats`` for the active lineup.
            lineup_probs: Probability matrix shape (9, 8). Row order matches ``players``.
            simulation_engine: ``MonteCarloEngine`` instance from simulation_engine.py.
            opp_lineup_probs: Opponent probability matrix, shape (9, 8). Constante
                entre órdenes; sirve para ordenar nuestra alineación (y, con
                ``objective="win_probability"``, para la P(W) del refinamiento).
            config: GA configuration; uses defaults if not provided.

        Nota (audit A1): el modelado del bullpen rival en E[R] NO está
        implementado. La capa de fitness usa un único perfil de pitcher (el
        abridor, codificado en ``lineup_probs``) para las 9 entradas. El antiguo
        par ``opp_bullpen_probs``/``bullpen_weight`` era un no-op —
        ``run_fast`` devuelve las carreras de MI lineup, independientes del
        oponente— y se ha retirado para no comunicar una capacidad inexistente.
        Modelar el relevo de verdad (prob_vectors del bateador vs bullpen +
        simulación por-entrada de mi ofensiva) es un ítem de roadmap (Fase C).

        Raises:
            ValueError: If player count != 9 or probability shapes are wrong.
        """
        if len(players) != 9:
            raise ValueError(f"Expected 9 players, got {len(players)}.")
        if lineup_probs.shape != (9, N_OUTCOMES):
            raise ValueError(
                f"lineup_probs must be (9, {N_OUTCOMES}), got {lineup_probs.shape}."
            )
        if opp_lineup_probs.shape != (9, N_OUTCOMES):
            raise ValueError(
                f"opp_lineup_probs must be (9, {N_OUTCOMES}), got {opp_lineup_probs.shape}."
            )

        self.players          = players
        self.lineup_probs     = lineup_probs.astype(np.float32)
        self.opp_lineup_probs = opp_lineup_probs.astype(np.float32)
        self.sim              = simulation_engine
        self.config           = config or GAConfig()
        self._seeder          = SabermetricSeeder(players)
        self._n_evals: int    = 0

        # DEAP toolbox (registered once in __init__)
        self._toolbox = self._create_toolbox()

    # ------------------------------------------------------------------
    # DEAP toolbox setup
    # ------------------------------------------------------------------

    def _create_toolbox(self) -> base.Toolbox:
        """Registers all DEAP genetic operators and the fitness function.

        DEAP uses a ``creator`` / ``toolbox`` pattern:
        - ``creator`` defines the Individual and Fitness types at class level.
        - ``toolbox`` registers the specific operations to call on them.

        Operators registered:
            - ``toolbox.individual``: factory for one 9-element permutation.
            - ``toolbox.population``: factory for a list of individuals.
            - ``toolbox.evaluate``: fitness function → returns (E[R],).
            - ``toolbox.mate``: cxOrdered crossover.
            - ``toolbox.mutate``: swap mutation.
            - ``toolbox.select``: tournament selection (k=3).

        Returns:
            Configured ``base.Toolbox`` instance.
        """
        # Guard: DEAP creator uses module-level class registration;
        # check before re-registering to avoid RuntimeWarning on re-import.
        if not hasattr(creator, "FitnessMaxER"):
            creator.create("FitnessMaxER", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "LineupIndividual"):
            creator.create("LineupIndividual", list, fitness=creator.FitnessMaxER)

        toolbox = base.Toolbox()

        # Individual factory: random permutation of [0..8]
        toolbox.register(
            "indices",
            random.sample,
            population=list(range(9)),
            k=9,
        )
        toolbox.register(
            "individual",
            tools.initIterate,
            creator.LineupIndividual,
            toolbox.indices,
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Fitness: E[R] from fast Monte Carlo
        toolbox.register("evaluate", self._evaluate_fitness)

        # Ordered Crossover (permutation-safe)
        toolbox.register("mate", cxOrdered)

        # Swap mutation (permutation-safe)
        toolbox.register("mutate", mutSwapPositions, indpb=self.config.mut_indpb)

        # Tournament selection
        toolbox.register("select", tools.selTournament, tournsize=self.config.tournament_size)

        return toolbox

    # ------------------------------------------------------------------
    # Fitness function
    # ------------------------------------------------------------------

    def _evaluate_fitness(self, individual: list[int]) -> tuple[float]:
        """Computes E[R] for a candidate batting order using fast simulation.

        Reorders the pre-computed probability matrix according to the batting
        order permutation, then runs a 5,000-simulation fast-mode evaluation.

        The ``opp_lineup_probs`` matrix is held constant across all fitness
        evaluations — we optimize our lineup given the opponent is fixed.

        Args:
            individual: Permutation of player indices [0–8] representing
                the batting order (position 0 = leadoff, 8 = ninth).

        Returns:
            Tuple ``(expected_runs,)`` (DEAP requires a tuple return).
        """
        ordered_probs = self.lineup_probs[individual].astype(np.float32)
        er = self.sim.run_fast(ordered_probs, self.opp_lineup_probs)
        self._n_evals += 1
        return (er,)

    # ------------------------------------------------------------------
    # Layer 1: Sabermetric population initialization
    # ------------------------------------------------------------------

    def _initialize_population(self, pitcher_hand: str = "R") -> list:
        """Creates the initial population mixing sabermetric seeds and random individuals.

        Composition:
        - First ``config.n_sabermetric_seeds`` individuals from the sabermetric seeder.
        - Remaining ``config.population_size - n_sabermetric_seeds`` random permutations.

        Args:
            pitcher_hand: Opposing starter's throwing hand; used for platoon seeding.

        Returns:
            List of ``config.population_size`` DEAP individuals.
        """
        cfg = self.config
        population: list = []

        # Sabermetric seeds
        seeds = self._seeder.diverse_seeds(cfg.n_sabermetric_seeds, pitcher_hand)
        for seed in seeds:
            ind = creator.LineupIndividual(seed)
            population.append(ind)

        # Random individuals for diversity
        n_random = cfg.population_size - cfg.n_sabermetric_seeds
        random_pop = self._toolbox.population(n=n_random)
        population.extend(random_pop)

        log.info(
            "population_initialized",
            total=len(population),
            sabermetric_seeds=cfg.n_sabermetric_seeds,
            random_individuals=n_random,
        )
        return population

    # ------------------------------------------------------------------
    # Layer 2: Genetic Algorithm main loop
    # ------------------------------------------------------------------

    def _run_ga(
        self,
        population: list,
    ) -> tuple[list, list[float], list[float]]:
        """Executes the genetic algorithm evolution loop.

        Implements a standard (μ + λ) evolutionary strategy with elitism:
        - The top ``config.elite_size`` individuals are preserved unchanged.
        - The rest are replaced by offspring from crossover and mutation.
        - Statistics (best and mean E[R]) are tracked per generation.

        Args:
            population: Initial population from Layer 1.

        Returns:
            Tuple ``(final_population, best_er_history, avg_er_history)``.
        """
        cfg = self.config
        toolbox = self._toolbox
        stats = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats.register("max",  np.max)
        stats.register("mean", np.mean)
        stats.register("std",  np.std)

        hof = tools.HallOfFame(maxsize=cfg.refinement_top_k)
        best_history: list[float] = []
        avg_history:  list[float] = []

        # Evaluate the initial population
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        hof.update(population)
        record = stats.compile(population)
        best_history.append(float(record["max"]))
        avg_history.append(float(record["mean"]))
        log.info("ga_gen_0", max_er=round(record["max"], 4), mean_er=round(record["mean"], 4))

        for gen in range(1, cfg.n_generations + 1):
            # Elitism: preserve top-N unchanged
            elite = tools.selBest(population, cfg.elite_size)
            elite = list(map(toolbox.clone, elite))

            # Tournament selection for the rest
            offspring = toolbox.select(population, cfg.population_size - cfg.elite_size)
            offspring = list(map(toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cfg.cx_probability:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < cfg.mut_probability:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Re-evaluate individuals with invalid fitness
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(toolbox.evaluate, invalid))
            for ind, fit in zip(invalid, fitnesses):
                ind.fitness.values = fit

            # Combine elite + offspring
            population[:] = elite + offspring
            hof.update(population)

            record = stats.compile(population)
            best_history.append(float(record["max"]))
            avg_history.append(float(record["mean"]))

            if gen % 25 == 0 or gen == cfg.n_generations:
                log.info(
                    "ga_generation",
                    gen=gen,
                    max_er=round(record["max"], 4),
                    mean_er=round(record["mean"], 4),
                    std_er=round(record["std"], 4),
                    n_evals=self._n_evals,
                )

        return list(hof), best_history, avg_history

    # ------------------------------------------------------------------
    # Layer 3: High-fidelity refinement
    # ------------------------------------------------------------------

    def _refine_top_k(
        self,
        candidates: list[list[int]],
    ) -> tuple[list[int], float]:
        """Evaluates top-K GA candidates with the full 100k-simulation engine.

        Replaces the noisy fast-proxy fitness values with high-confidence E[R]
        estimates. The variance of the 5,000-sim proxy is ~0.05 runs, which
        can cause the GA to converge to a sub-optimal solution. The full engine
        reduces this to ~0.01 runs, reliably identifying the true best lineup.

        Args:
            candidates: Top-K permutation lists from the GA Hall of Fame.

        Returns:
            Tuple ``(best_lineup_indices, best_er_from_full_simulation)``.
        """
        objective = getattr(self.config, "objective", "expected_runs")

        best_metric = -np.inf
        best_er   = -np.inf
        best_se   = 0.0
        best_idx  = candidates[0]
        best_win  = 0.5
        scores: list[float] = []        # E[R] por candidato (= refinement_scores)
        win_scores: list[float] = []    # P(W) por candidato
        er_ses: list[float] = []        # SE de muestreo del E[R]
        metric_ses: list[float] = []    # SE de la métrica OBJETIVO (E[R] o P(W))

        for i, cand in enumerate(candidates):
            ordered_probs = self.lineup_probs[cand].astype(np.float32)
            result = self.sim.run(
                ordered_probs,
                self.opp_lineup_probs,
                fast_mode=False,
            )
            er  = result.expected_runs_scored
            win = result.win_probability
            # Error estándar del E[R] estimado por Monte Carlo (audit F18):
            # SE = sigma_runs / sqrt(n_sims). Cuantifica el ruido de muestreo.
            er_se = result.std_dev_runs_scored / max(np.sqrt(result.n_simulations), 1.0)
            # SE binomial de la win probability (audit A2): sqrt(p(1-p)/n).
            win_se = float(np.sqrt(max(win * (1.0 - win), 0.0)
                                   / max(result.n_simulations, 1)))
            scores.append(er)
            win_scores.append(win)
            er_ses.append(float(er_se))

            # Métrica que decide el ganador, según el objetivo configurado.
            metric    = win if objective == "win_probability" else er
            metric_se = win_se if objective == "win_probability" else float(er_se)
            metric_ses.append(metric_se)

            log.info(
                "refinement_eval",
                rank=i + 1,
                objective=objective,
                er=round(er, 4),
                er_se=round(float(er_se), 4),
                win_prob=round(win, 4),
                win_se=round(win_se, 4),
                lineup=[self.players[j].player_name for j in cand],
            )

            if metric > best_metric:
                best_metric = metric
                best_er  = er
                best_se  = float(er_se)
                best_idx = cand
                best_win = win

        # ── Test de significancia del ganador vs el runner-up (audit F18/A2) ───
        # Sobre la MÉTRICA OBJETIVO: si los dos mejores solapan dentro de su error
        # de muestreo conjunto, la elección es ruido y se declara explícitamente
        # en vez de fingir que el orden #1 es estrictamente mejor.
        primary = win_scores if objective == "win_probability" else scores
        significant = True
        delta_vs_second = 0.0
        if len(primary) >= 2:
            order = np.argsort(primary)[::-1]
            i1, i2 = int(order[0]), int(order[1])
            delta_vs_second = float(primary[i1] - primary[i2])
            se_joint = float(np.sqrt(metric_ses[i1] ** 2 + metric_ses[i2] ** 2))
            # Significativo al 90% si |delta| > 1.645 * SE_conjunto
            significant = delta_vs_second > 1.645 * se_joint

        log.info(
            "refinement_complete",
            objective=objective,
            best_er=round(best_er, 4),
            best_er_se=round(best_se, 4),
            best_win=round(best_win, 4),
            delta_vs_second=round(delta_vs_second, 4),
            significant_at_90=significant,
        )
        if not significant:
            log.warning(
                "lineup_choice_not_significant",
                msg=("El mejor orden no supera al runner-up más allá del ruido MC. "
                     "Considera aumentar refinement_n_sims o tratar ambos como equivalentes."),
                delta_vs_second=round(delta_vs_second, 4),
            )
        return best_idx, best_er, best_win, scores, best_se, significant

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        pitcher_hand: str = "R",
        park_factor_hr: float = 1.0,
        park_factor_xb: float = 1.0,
    ) -> OptimizerResult:
        """Executes the full 3-layer lineup optimization.

        Layer 1 → sabermetric seeding
        Layer 2 → genetic algorithm (150 generations × 200 individuals)
        Layer 3 → high-fidelity refinement of top 10

        Args:
            pitcher_hand: Opposing starter's throwing hand (``"L"`` or ``"R"``).
                Used for platoon-aware sabermetric seeding.
            park_factor_hr: HR park factor for today's venue.
            park_factor_xb: Extra-base hit park factor.

        Returns:
            ``OptimizerResult`` with the optimal lineup and full diagnostics.
        """
        cfg = self.config
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        self._n_evals = 0

        # Propagate park factors to the simulation engine config
        self.sim.config.park_factor_hr = park_factor_hr
        self.sim.config.park_factor_xb = park_factor_xb

        t_start = time.perf_counter()

        mlflow.set_experiment(cfg.mlflow_experiment)
        with mlflow.start_run(run_name=f"ga-optimizer-{time.strftime('%Y%m%d-%H%M%S')}"):  # noqa: SIM117
            mlflow.log_params({
                "population_size":    cfg.population_size,
                "n_generations":      cfg.n_generations,
                "cx_probability":     cfg.cx_probability,
                "mut_probability":    cfg.mut_probability,
                "tournament_size":    cfg.tournament_size,
                "elite_size":         cfg.elite_size,
                "fast_mode_n_sims":   cfg.fast_mode_n_sims,
                "refinement_n_sims":  cfg.refinement_n_sims,
                "refinement_top_k":   cfg.refinement_top_k,
                "park_factor_hr":     park_factor_hr,
                "pitcher_hand":       pitcher_hand,
            })

            # Layer 1: seeded population
            log.info("optimizer_layer1_start")
            population = self._initialize_population(pitcher_hand)

            # Layer 2: genetic algorithm
            log.info("optimizer_layer2_start", generations=cfg.n_generations)
            top_candidates, best_hist, avg_hist = self._run_ga(population)

            # Log per-generation statistics to MLflow
            for gen_i, (b, a) in enumerate(zip(best_hist, avg_hist)):
                mlflow.log_metrics({"ga_best_er": b, "ga_avg_er": a}, step=gen_i)

            ga_elapsed = time.perf_counter() - t_start
            log.info(
                "optimizer_layer2_complete",
                elapsed_s=round(ga_elapsed, 1),
                n_evals=self._n_evals,
                top_candidate_er=round(max(ind.fitness.values[0] for ind in top_candidates), 4),
            )

            # Layer 3: high-fidelity refinement
            log.info("optimizer_layer3_start", n_candidates=len(top_candidates))
            candidate_permutations = [list(ind) for ind in top_candidates]
            (best_order, best_er, best_win, ref_scores,
             best_er_se, is_significant) = self._refine_top_k(candidate_permutations)

            total_elapsed = time.perf_counter() - t_start
            log.info(
                "optimizer_complete",
                total_elapsed_s=round(total_elapsed, 1),
                best_er=round(best_er, 4),
                best_win=round(best_win, 4),
                total_evals=self._n_evals,
            )

            mlflow.log_metrics({
                "final_best_er":   best_er,
                "final_best_win":  best_win,
                "total_elapsed_s": total_elapsed,
                "n_fitness_evals": self._n_evals,
            })

        result = OptimizerResult(
            best_lineup_indices=best_order,
            best_lineup_ids=[self.players[i].player_id for i in best_order],
            best_lineup_names=[self.players[i].player_name for i in best_order],
            best_expected_runs=best_er,
            best_win_probability=best_win,
            ga_best_er_history=best_hist,
            ga_avg_er_history=avg_hist,
            refinement_scores=ref_scores,
            total_elapsed_seconds=total_elapsed,
            n_fitness_evaluations=self._n_evals,
            best_expected_runs_se=best_er_se,
            is_significant_vs_second=is_significant,
        )
        return result


# ---------------------------------------------------------------------------
# Convenience function for one-shot optimization
# ---------------------------------------------------------------------------

def optimize_lineup(
    players: list[PlayerStats],
    lineup_probs: np.ndarray,
    opp_lineup_probs: np.ndarray,
    simulation_engine,
    park_factor_hr: float = 1.0,
    park_factor_xb: float = 1.0,
    pitcher_hand: str = "R",
    config: Optional[GAConfig] = None,
) -> OptimizerResult:
    """Convenience wrapper: builds the optimizer and runs a full optimization.

    Args:
        players: List of 9 ``PlayerStats`` for today's lineup.
        lineup_probs: My team's probability matrix, shape (9, 8).
        opp_lineup_probs: Opponent's probability matrix, shape (9, 8).
        simulation_engine: Configured ``MonteCarloEngine`` instance.
        park_factor_hr: HR park factor for today's venue.
        park_factor_xb: Extra-base hit park factor.
        pitcher_hand: Opposing starter's hand (``"L"`` or ``"R"``).
        config: Optional ``GAConfig``; uses defaults if not provided.

    Returns:
        ``OptimizerResult`` with the optimal lineup.
    """
    optimizer = GeneticLineupOptimizer(
        players=players,
        lineup_probs=lineup_probs,
        simulation_engine=simulation_engine,
        opp_lineup_probs=opp_lineup_probs,
        config=config,
    )
    return optimizer.run(
        pitcher_hand=pitcher_hand,
        park_factor_hr=park_factor_hr,
        park_factor_xb=park_factor_xb,
    )
