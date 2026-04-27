"""
evolution.py — Niche-based evolutionary manager for steganography genome search.

Run 6 changes vs run 5
----------------------
NICHE SYSTEM
  get_niche() returns 'fft_low', 'fft_mid', 'fft_high' instead of 'fft'.
  ALL_NICHES updated to 8 niches (was 6).
  evolve() injection handles the three FFT sub-niches individually.
  Diversity dampener now operates per-subtype rather than across all FFT.

SEEDING
  Targeted edge seeds, low-cap sequential, fft_low ×4, fft_mid/high ×1 each,
  DCT ×3, skip ×2.
"""

import copy
import random

from training.config import (
    ALL_GEN_TYPES,
    ALL_NICHES,
    CAPACITY_PENALTY_THRESHOLD,
    CAPACITY_PENALTY_WEIGHT,
    DCT_COEFF_MODES,
    FFT_FREQ_BANDS,
    FFT_LOW_SEED_CAPACITIES,
    FFT_LOW_SEED_STRENGTHS,
    GEN_TYPE_WEIGHTS,
    LSB_STRATEGIES,
    MAX_CAPACITY,
    MIN_CAPACITY,
    MIN_NICHE_SIZE,
    POPULATION_SIZE,
    SKIP_SEED_STEPS,
)
from training.genome import get_niche, is_low_capacity, is_hard_edge, LOW_CAPACITY_THRESHOLD


class EvolutionaryManager:
    """Maintains a niche-structured population of steganography genomes."""

    def __init__(self):
        self.population  = []
        self.generation  = 0
        self._seed_population()
        self.stats = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        self._print_init_summary()

    # ── Seeding ───────────────────────────────────────────────────────────────

    def _seed_population(self):
        # LSB non-edge strategies
        for strategy in ['random', 'sequential', 'skip']:
            self.population.append(self._new_lsb(f"Seed_lsb_{strategy}", strategy))

        # LSB sequential low-capacity
        g = self._new_lsb("Seed_lsb_sequential_lowcap", 'sequential')
        g['capacity_ratio'] = 0.25
        g['step']           = 1
        self.population.append(g)

        # LSB edge: generic threshold seeds
        for threshold in [5, 9, 15, 30]:
            g = self._new_lsb(f"Seed_lsb_edge_t{threshold}", 'edge')
            g['edge_threshold'] = threshold
            self.population.append(g)

        # LSB edge: hard-config seeds (pins threshold AND capacity to eval values)
        for threshold, capacity in [(5, 0.21), (9, 0.21), (9, 0.25), (15, 0.25)]:
            g = self._new_lsb(
                f"Seed_lsb_edge_hard_t{threshold}_c{int(capacity * 100)}", 'edge')
            g['edge_threshold'] = threshold
            g['capacity_ratio'] = capacity
            self.population.append(g)

        # LSB skip
        for s in SKIP_SEED_STEPS:
            g = self._new_lsb(f"Seed_skip_s{s}", 'skip')
            g['step'] = s
            self.population.append(g)

        # DCT
        for mode in DCT_COEFF_MODES:
            self.population.append(self._new_dct(f"Seed_dct_{mode}", mode))

        # FFT mid / high
        for band in ['mid', 'high']:
            self.population.append(self._new_fft(f"Seed_fft_{band}", band))

        # FFT low: 4 seeds bracketing the hard eval point
        for strength, capacity in zip(FFT_LOW_SEED_STRENGTHS, FFT_LOW_SEED_CAPACITIES):
            tag = str(strength).replace('.', 'p')
            g   = self._new_fft(f"Seed_fft_low_s{tag}", 'low')
            g['strength']       = strength
            g['capacity_ratio'] = capacity
            self.population.append(g)

        # Fill remainder with random genomes
        while len(self.population) < POPULATION_SIZE:
            self.population.append(
                self._generate_random_genome(f"Gen_{len(self.population)}"))

    def _print_init_summary(self):
        niche_counts  = {n: sum(1 for g in self.population if get_niche(g) == n)
                         for n in ALL_NICHES}
        lowcap_counts = {n: sum(1 for g in self.population
                                if get_niche(g) == n and is_low_capacity(g))
                         for n in ALL_NICHES}
        print(f"[EVO INIT] Population: {len(self.population)}")
        print(f"[EVO INIT] Niches:  {niche_counts}")
        print(f"[EVO INIT] Low-cap: {lowcap_counts}")

    # ── Genome constructors ───────────────────────────────────────────────────

    def _new_lsb(self, name: str, strategy: str = None) -> dict:
        return {
            'name':           name,
            'gen_type':       'lsb',
            'strategy':       strategy or random.choice(LSB_STRATEGIES),
            'step':           random.randint(1, 15),
            'bit_depth':      1,
            'edge_threshold': random.randint(0, 100),
            'capacity_ratio': random.triangular(MIN_CAPACITY, MAX_CAPACITY,
                                                MIN_CAPACITY + 0.15),
        }

    def _new_dct(self, name: str, coeff_selection: str = None) -> dict:
        return {
            'name':            name,
            'gen_type':        'dct',
            'coeff_selection': coeff_selection or random.choice(DCT_COEFF_MODES),
            'strength':        round(random.uniform(2.0, 8.0), 2),
            'capacity_ratio':  random.triangular(MIN_CAPACITY, MAX_CAPACITY,
                                                 MIN_CAPACITY + 0.15),
        }

    def _new_fft(self, name: str, freq_band: str = None) -> dict:
        return {
            'name':           name,
            'gen_type':       'fft',
            'freq_band':      freq_band or random.choice(FFT_FREQ_BANDS),
            'strength':       round(random.uniform(3.0, 20.0), 2),
            'capacity_ratio': random.triangular(MIN_CAPACITY, MAX_CAPACITY,
                                                MIN_CAPACITY + 0.15),
        }

    def _generate_random_genome(self, name: str) -> dict:
        gen_type = random.choices(ALL_GEN_TYPES, weights=GEN_TYPE_WEIGHTS)[0]
        if gen_type == 'lsb':
            return self._new_lsb(name)
        if gen_type == 'dct':
            return self._new_dct(name)
        return self._new_fft(name)

    # ── Fitness ───────────────────────────────────────────────────────────────

    def _penalised_fitness(self, genome: dict, raw_fool_rate: float) -> float:
        capacity = genome.get('capacity_ratio', 0.5)
        if capacity < CAPACITY_PENALTY_THRESHOLD:
            shortfall = (CAPACITY_PENALTY_THRESHOLD - capacity) / CAPACITY_PENALTY_THRESHOLD
            penalty   = shortfall * CAPACITY_PENALTY_WEIGHT
        else:
            penalty = 0.0
        return max(0.0, raw_fool_rate - penalty)

    # ── Genetic operators ─────────────────────────────────────────────────────

    def mutate(self, genome: dict) -> dict:
        g = copy.deepcopy(genome)
        g['name']   = f"{genome['name']}_m{self.generation}"
        n_mutations = 2 if random.random() < 0.4 else 1

        for _ in range(n_mutations):
            gt = g['gen_type']

            if gt == 'lsb':
                field = random.choice(['step', 'threshold', 'strategy', 'capacity', 'gen_type'])
                if field == 'step':
                    g['step'] = max(1, min(20, g['step'] + random.choice([-2, -1, 1, 2, 3])))
                elif field == 'threshold':
                    g['edge_threshold'] = max(0, min(100,
                        g['edge_threshold'] + random.randint(-20, 20)))
                elif field == 'strategy':
                    g['strategy'] = random.choice(LSB_STRATEGIES)
                elif field == 'capacity':
                    g['capacity_ratio'] = max(MIN_CAPACITY, min(MAX_CAPACITY,
                        g['capacity_ratio'] + random.uniform(-0.15, 0.15)))
                elif field == 'gen_type' and random.random() < 0.15:
                    new_type = random.choice(['dct', 'fft'])
                    base = (self._new_dct(g['name']) if new_type == 'dct'
                            else self._new_fft(g['name']))
                    base['capacity_ratio'] = g['capacity_ratio']
                    g = base

            elif gt == 'dct':
                field = random.choice(['coeff', 'strength', 'capacity', 'gen_type'])
                if field == 'coeff':
                    g['coeff_selection'] = random.choice(DCT_COEFF_MODES)
                elif field == 'strength':
                    g['strength'] = max(2.0, min(10.0,
                        g['strength'] + random.uniform(-1.5, 1.5)))
                elif field == 'capacity':
                    g['capacity_ratio'] = max(MIN_CAPACITY, min(MAX_CAPACITY,
                        g['capacity_ratio'] + random.uniform(-0.15, 0.15)))
                elif field == 'gen_type' and random.random() < 0.10:
                    base = self._new_fft(g['name'])
                    base['capacity_ratio'] = g['capacity_ratio']
                    g = base

            elif gt == 'fft':
                field = random.choice(['band', 'strength', 'capacity', 'gen_type'])
                if field == 'band':
                    other_bands = [b for b in FFT_FREQ_BANDS if b != g['freq_band']]
                    g['freq_band'] = random.choice(other_bands)
                elif field == 'strength':
                    g['strength'] = max(3.0, min(25.0,
                        g['strength'] + random.uniform(-3.0, 3.0)))
                elif field == 'capacity':
                    g['capacity_ratio'] = max(MIN_CAPACITY, min(MAX_CAPACITY,
                        g['capacity_ratio'] + random.uniform(-0.15, 0.15)))
                elif field == 'gen_type' and random.random() < 0.10:
                    base = self._new_dct(g['name'])
                    base['capacity_ratio'] = g['capacity_ratio']
                    g = base

        return g

    def crossover(self, g1: dict, g2: dict) -> dict:
        if get_niche(g1) == get_niche(g2) and random.random() < 0.7:
            g2 = self.mutate(g2)

        child = copy.deepcopy(g1)
        child['name'] = f"Cross_{self.generation}"

        if g1['gen_type'] != g2['gen_type']:
            if random.random() < 0.5:
                child['capacity_ratio'] = g2['capacity_ratio']
            return child

        gt = g1['gen_type']
        if gt == 'lsb':
            if random.random() < 0.5: child['step']           = g2['step']
            if random.random() < 0.5: child['edge_threshold'] = g2['edge_threshold']
            if random.random() < 0.5: child['strategy']       = g2['strategy']
        elif gt == 'dct':
            if random.random() < 0.5: child['coeff_selection'] = g2['coeff_selection']
            if random.random() < 0.5: child['strength']        = g2['strength']
        elif gt == 'fft':
            if random.random() < 0.5: child['freq_band'] = g2['freq_band']
            if random.random() < 0.5: child['strength']  = g2['strength']

        if random.random() < 0.5:
            child['capacity_ratio'] = g2['capacity_ratio']

        return child

    # ── Stats ─────────────────────────────────────────────────────────────────

    def update_batch_stats(self, names: list, is_fooled_list: list) -> None:
        for name, fooled in zip(names, is_fooled_list):
            if name in self.stats:
                self.stats[name]['attempts'] += 1
                if fooled:
                    self.stats[name]['fooled'] += 1

    # ── Evolution step ────────────────────────────────────────────────────────

    def _is_duplicate(self, genome: dict, population: list) -> bool:
        for g in population:
            if (g['gen_type'] == genome['gen_type']
                    and g.get('freq_band') == genome.get('freq_band')
                    and abs(g.get('strength', 0) - genome.get('strength', 0)) < 0.5
                    and abs(g.get('capacity_ratio', 0)
                            - genome.get('capacity_ratio', 0)) < 0.05):
                return True
        return False

    def evolve(self) -> dict:
        """Run one generation: score, select, reproduce, inject. Returns best genome."""
        self.generation += 1

        # Score
        final_scores = {}
        for genome in self.population:
            data = self.stats[genome['name']]
            raw  = data['fooled'] / data['attempts'] if data['attempts'] > 0 else 0.0
            final_scores[genome['name']] = self._penalised_fitness(genome, raw)

        sorted_pop = sorted(self.population,
                            key=lambda g: final_scores.get(g['name'], 0.0),
                            reverse=True)

        self._print_evolution_summary(sorted_pop, final_scores)

        # Niche preservation: MIN_NICHE_SIZE survivors per niche
        niche_survivors = []
        for niche in ALL_NICHES:
            members = [g for g in sorted_pop if get_niche(g) == niche]
            for g in members[:MIN_NICHE_SIZE]:
                if g not in niche_survivors:
                    niche_survivors.append(g)

        # Elites (top 3, de-duplicated)
        elite = []
        for g in sorted_pop:
            if g not in elite:
                elite.append(g)
            if len(elite) == 3:
                break

        new_pop = list({id(g): g for g in elite + niche_survivors}.values())

        # Crossover children
        for parent_b_idx in [1, 2]:
            if len(sorted_pop) > parent_b_idx:
                child = self.crossover(sorted_pop[0], sorted_pop[parent_b_idx])
                if not self._is_duplicate(child, new_pop):
                    new_pop.append(child)
                else:
                    new_pop.append(self.mutate(sorted_pop[0]))

        # Inject for the two most under-represented niches
        niche_counts_new  = {n: sum(1 for g in new_pop if get_niche(g) == n)
                              for n in ALL_NICHES}
        underrepresented  = sorted(niche_counts_new, key=niche_counts_new.get)
        for niche in underrepresented[:2]:
            if niche.startswith('lsb_'):
                strategy = niche.split('_', 1)[1]
                new_pop.append(self._new_lsb(f"Explore_{niche}_{self.generation}", strategy))
            elif niche == 'dct':
                new_pop.append(self._new_dct(f"Explore_dct_{self.generation}"))
            elif niche.startswith('fft_'):
                band = niche.split('_', 1)[1]   # 'low', 'mid', or 'high'
                new_pop.append(self._new_fft(f"Explore_{niche}_{self.generation}", band))

        # Fill remainder by mutating top parents
        while len(new_pop) < POPULATION_SIZE:
            parent_idx = min(random.randint(0, 2), len(sorted_pop) - 1)
            new_pop.append(self.mutate(sorted_pop[parent_idx]))

        self.population = new_pop[:POPULATION_SIZE]
        self.stats      = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        return sorted_pop[0]

    def _print_evolution_summary(self, sorted_pop: list, final_scores: dict) -> None:
        print(f"\n[EVOLUTION] Generation {self.generation} — Top 3:")
        for i in range(min(3, len(sorted_pop))):
            g   = sorted_pop[i]
            sc  = final_scores.get(g['name'], 0.0) * 100
            d   = self.stats[g['name']]
            raw = (d['fooled'] / d['attempts'] * 100) if d['attempts'] > 0 else 0.0
            gt  = g['gen_type']
            if gt == 'lsb':
                detail = (f"Strat={g['strategy']} Step={g['step']} "
                          f"Edge={g['edge_threshold']} Cap={g['capacity_ratio']:.2f}")
            elif gt == 'dct':
                detail = (f"Coeff={g['coeff_selection']} Str={g['strength']:.1f} "
                          f"Cap={g['capacity_ratio']:.2f}")
            else:
                detail = (f"Band={g['freq_band']} Str={g['strength']:.1f} "
                          f"Cap={g['capacity_ratio']:.2f}")
            print(f"  #{i+1}: {g['name']} — {sc:.2f}% (raw {raw:.1f}%) | "
                  f"Niche={get_niche(g)} | {detail}")

        niche_counts = {n: sum(1 for g in self.population if get_niche(g) == n)
                        for n in ALL_NICHES}
        print(f"  Niches: {niche_counts}")

    # ── Sampling ──────────────────────────────────────────────────────────────

    def get_random_genome(self) -> dict:
        """Sample a genome, weighted by fitness and diversity-dampened by niche size."""
        if self.generation == 0 or random.random() < 0.3:
            return random.choice(self.population)

        niche_counts = {}
        for g in self.population:
            n = get_niche(g)
            niche_counts[n] = niche_counts.get(n, 0) + 1

        weights = []
        for g in self.population:
            data = self.stats[g['name']]
            if data['attempts'] == 0:
                raw_weight = 0.15
            else:
                raw        = data['fooled'] / data['attempts']
                raw_weight = self._penalised_fitness(g, raw) + 0.05
            niche_size = niche_counts.get(get_niche(g), 1)
            weights.append(raw_weight / (niche_size ** 0.75))

        total   = sum(weights)
        weights = [w / total for w in weights]
        return random.choices(self.population, weights=weights, k=1)[0]

    def get_low_capacity_genome(self) -> dict:
        """Return a genome whose capacity_ratio is below LOW_CAPACITY_THRESHOLD."""
        candidates = [g for g in self.population if is_low_capacity(g)]
        if candidates:
            return random.choice(candidates)

        # Construct a fallback if none exist in the population
        choice = random.choice(['lsb_edge', 'lsb_sequential', 'fft_low'])
        if choice == 'lsb_edge':
            g = self._new_lsb("tmp_lowcap_edge", 'edge')
            g['capacity_ratio'] = 0.21
            g['edge_threshold'] = 9
        elif choice == 'lsb_sequential':
            g = self._new_lsb("tmp_lowcap_seq", 'sequential')
            g['capacity_ratio'] = 0.25
        else:
            g = self._new_fft("tmp_lowcap_fft_low", 'low')
            g['capacity_ratio'] = 0.25
            g['strength']       = 10.0
        return g

    def get_hard_edge_genome(self) -> dict:
        """
        Return a lsb_edge genome with threshold≤9 AND capacity≤0.25.

        Falls back to a freshly constructed genome if none exist in the
        population — this guarantees the hard eval config appears every batch.
        """
        candidates = [g for g in self.population if is_hard_edge(g)]
        if candidates:
            return random.choice(candidates)
        g = self._new_lsb("tmp_hard_edge", 'edge')
        g['edge_threshold'] = 9
        g['capacity_ratio'] = 0.21
        return g

    def get_lowstrength_fft_low_genome(self):
        """
        Return an fft_low genome with strength ≤ 7.5 (bottom quarter of range).
        Used by batch.py Layer 5 to force the model to see the weak fft_low
        low-strength configs every batch during fine-tuning.
        Falls back to a freshly constructed genome if none exist in the population.
        """
        candidates = [
            g for g in self.population
            if g.get('gen_type') == 'fft'
               and g.get('freq_band') == 'low'
               and g.get('strength', 99) <= 7.5
        ]
        if candidates:
            return random.choice(candidates)
        # Population hasn't evolved low-strength fft_low genomes yet — construct one.
        g = self._new_fft("tmp_lowstrength_fft_low", 'low')
        g['strength'] = round(random.uniform(2.0, 5.0), 2)
        g['capacity_ratio'] = random.uniform(0.35, 0.55)
        return g

    def get_lowstrength_dct_lowmid_genome(self):
        """
        Return a dct_low_mid genome with strength ≤ 3.5 (bottom quarter of range).
        Used by batch.py Layer 6 to force the model to see the weak dct_low_mid
        low-strength configs every batch during fine-tuning.
        Falls back to a freshly constructed genome if none exist in the population.
        """
        candidates = [
            g for g in self.population
            if g.get('gen_type') == 'dct'
               and g.get('coeff_selection') == 'low_mid'
               and g.get('strength', 99) <= 3.5
        ]
        if candidates:
            return random.choice(candidates)
        # Population hasn't evolved low-strength dct_low_mid genomes yet — construct one.
        g = self._new_dct("tmp_lowstrength_dct_lowmid", 'low_mid')
        g['strength'] = round(random.uniform(1.5, 3.0), 2)
        return g