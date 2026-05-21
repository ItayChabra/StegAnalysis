"""
evolution.py — Niche-based evolutionary manager for steganography genome search.

Scope: cover-vs-stego across {LSB, DCT, FFT, S-UNIWARD}. WOW, HUGO and the LSB
'edge' strategy were dropped — low-payload adaptive is statistically
undetectable and acted as label noise; S-UNIWARD is trained at high payload.

CAPACITY SEMANTICS
  capacity_ratio is TRUE bits-per-pixel for every gen_type. Per-method ranges
  (LSB/DCT/FFT_CAPACITY_RANGE) and strength floors (DCT/FFT_STRENGTH_RANGE) in
  config.py fence the EA's search space to learnable, detectable configs — so
  the fool-rate fitness can no longer collapse onto a near-invisible corner.

NICHE SYSTEM
  get_niche() returns 'fft_low'/'fft_mid'/'fft_high' for FFT and
  'adaptive_suniward' for adaptive. See ALL_NICHES in config.py.

ADAPTIVE INTEGRATION
  Adaptive (S-UNIWARD) lives in a SEPARATE 4-genome sub-population. The EA
  evolves only its cost-model SHAPE (sigma_offset, cost_exponent, use_diagonal)
  — never the payload. The adaptive payload is set by the curriculum schedule
  at embed time (config.ADAPTIVE_CURRICULUM_SCHEDULE, applied in train_hybrid).
  Keeping the pools separate stops adaptive (often a high fool rate at low
  curriculum payloads) from crowding LSB/DCT/FFT out of the elite/crossover
  breeding. get_random_genome() returns non-adaptive genomes only — adaptive
  enters every batch via the Layer-7 floor (get_adaptive_genome).
"""

import copy
import random

from training.config import (
    ADAPTIVE_MODES,
    ADAPTIVE_SEED_CONFIGS,
    ALL_GEN_TYPES,
    ALL_NICHES,
    CAPACITY_PENALTY_THRESHOLDS,
    CAPACITY_PENALTY_WEIGHT,
    DCT_CAPACITY_RANGE,
    DCT_COEFF_MODES,
    DCT_STRENGTH_RANGE,
    FFT_CAPACITY_RANGE,
    FFT_FREQ_BANDS,
    FFT_LOW_SEED_CAPACITIES,
    FFT_LOW_SEED_STRENGTHS,
    FFT_STRENGTH_RANGE,
    GEN_TYPE_WEIGHTS,
    LSB_CAPACITY_RANGE,
    LSB_STRATEGIES,
    MIN_NICHE_SIZE,
    POPULATION_SIZE,
    SKIP_SEED_STEPS,
)
from training.genome import get_niche, is_low_capacity, LOW_CAPACITY_THRESHOLD

# Size of the adaptive shape-evolving sub-population (one slot per seed shape).
_ADAPTIVE_POP_SIZE = len(ADAPTIVE_SEED_CONFIGS)


class EvolutionaryManager:
    """Maintains a niche-structured population of steganography genomes."""

    def __init__(self):
        self.population     = []
        self.generation     = 0
        self._genome_counter = 0
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

        # FFT low: 4 seeds varying strength
        for strength, capacity in zip(FFT_LOW_SEED_STRENGTHS, FFT_LOW_SEED_CAPACITIES):
            tag = str(strength).replace('.', 'p')
            g   = self._new_fft(f"Seed_fft_low_s{tag}", 'low')
            g['strength']       = strength
            g['capacity_ratio'] = capacity
            self.population.append(g)

        # Adaptive: one genome per ADAPTIVE_SEED_CONFIGS entry (distinct shapes)
        for mode, sigma_off, cap, exp in ADAPTIVE_SEED_CONFIGS:
            tag = f"{mode}_s{str(sigma_off).replace('.', 'p')}"
            self.population.append(
                self._new_adaptive(f"Seed_adaptive_{tag}", mode, sigma_off, cap, exp))

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
        lo, hi = LSB_CAPACITY_RANGE
        return {
            'name':           name,
            'gen_type':       'lsb',
            'strategy':       strategy or random.choice(LSB_STRATEGIES),
            'step':           random.randint(1, 8),
            'bit_depth':      1,
            'capacity_ratio': random.triangular(lo, hi, lo + 0.15),
        }

    def _new_dct(self, name: str, coeff_selection: str = None) -> dict:
        lo, hi = DCT_CAPACITY_RANGE
        return {
            'name':            name,
            'gen_type':        'dct',
            'coeff_selection': coeff_selection or random.choice(DCT_COEFF_MODES),
            'strength':        round(random.uniform(*DCT_STRENGTH_RANGE), 2),
            'capacity_ratio':  random.triangular(lo, hi, hi - 0.08),
        }

    def _new_fft(self, name: str, freq_band: str = None) -> dict:
        lo, hi = FFT_CAPACITY_RANGE
        return {
            'name':           name,
            'gen_type':       'fft',
            'freq_band':      freq_band or random.choice(FFT_FREQ_BANDS),
            'strength':       round(random.uniform(*FFT_STRENGTH_RANGE), 2),
            'capacity_ratio': random.triangular(lo, hi, hi - 0.05),
        }

    def _new_adaptive(self, name: str, adaptive_mode: str = None,
                      sigma_offset: float = None, capacity_ratio: float = None,
                      cost_exponent: float = None) -> dict:
        # capacity_ratio is a PLACEHOLDER — the adaptive payload is set by the
        # curriculum schedule at embed time, never evolved. The EA evolves only
        # the cost-model shape: sigma_offset, cost_exponent, use_diagonal.
        return {
            'name':           name,
            'gen_type':       'adaptive',
            'adaptive_mode':  adaptive_mode or random.choice(ADAPTIVE_MODES),
            'sigma_offset':   (sigma_offset if sigma_offset is not None
                               else round(random.uniform(0.5, 3.0), 2)),
            'capacity_ratio': capacity_ratio if capacity_ratio is not None else 0.40,
            'cost_exponent':  (cost_exponent if cost_exponent is not None
                               else round(random.uniform(0.7, 1.5), 2)),
            'use_diagonal':   True,
            'canonical':      True,
        }

    def _generate_random_genome(self, name: str) -> dict:
        gen_type = random.choices(ALL_GEN_TYPES, weights=GEN_TYPE_WEIGHTS)[0]
        if gen_type == 'lsb':
            return self._new_lsb(name)
        if gen_type == 'dct':
            return self._new_dct(name)
        if gen_type == 'adaptive':
            return self._new_adaptive(name)
        return self._new_fft(name)

    # ── Fitness ───────────────────────────────────────────────────────────────

    def _penalised_fitness(self, genome: dict, raw_fool_rate: float) -> float:
        """
        Fitness = raw fool rate minus a capacity anti-collapse penalty.

        The penalty ramps linearly from 0 at the genome's per-method threshold
        (CAPACITY_PENALTY_THRESHOLDS) to CAPACITY_PENALTY_WEIGHT at capacity 0,
        so the EA cannot win by hiding in a near-undetectable low-capacity corner.
        Each method's threshold sits near the middle of its own bits-per-pixel
        range (a single absolute threshold is unfair across methods whose
        ceilings span ~0.017 → 1.0 bpp). Adaptive's threshold equals its floor,
        so adaptive — whose capacity is curriculum-set, not EA-evolved — is never
        penalised.
        """
        capacity  = genome.get('capacity_ratio', 0.5)
        gen_type  = genome.get('gen_type', 'lsb')
        threshold = CAPACITY_PENALTY_THRESHOLDS.get(gen_type, 0.30)
        if capacity < threshold:
            shortfall = (threshold - capacity) / threshold
            penalty   = shortfall * CAPACITY_PENALTY_WEIGHT
        else:
            penalty = 0.0
        return max(0.0, raw_fool_rate - penalty)

    # ── Genetic operators ─────────────────────────────────────────────────────

    def _fresh_id(self) -> int:
        """Monotonic counter so mutated / crossover genome names never collide
        within a generation — two children built in the same generation would
        otherwise share a name and merge their fitness in self.stats."""
        self._genome_counter += 1
        return self._genome_counter

    def mutate(self, genome: dict) -> dict:
        g = copy.deepcopy(genome)
        g['name']   = f"{genome['name']}_m{self.generation}n{self._fresh_id()}"
        n_mutations = 2 if random.random() < 0.4 else 1

        for _ in range(n_mutations):
            gt = g['gen_type']

            if gt == 'lsb':
                field = random.choice(['step', 'strategy', 'capacity', 'gen_type'])
                if field == 'step':
                    g['step'] = max(1, min(20, g['step'] + random.choice([-2, -1, 1, 2, 3])))
                    if g.get('strategy') == 'skip':
                        g['step'] = min(g['step'], 8)
                elif field == 'strategy':
                    g['strategy'] = random.choice(LSB_STRATEGIES)
                    if g['strategy'] == 'skip':
                        g['step'] = min(g.get('step', 8), 8)
                elif field == 'capacity':
                    g['capacity_ratio'] = max(LSB_CAPACITY_RANGE[0], min(LSB_CAPACITY_RANGE[1],
                        g['capacity_ratio'] + random.uniform(-0.15, 0.15)))
                elif field == 'gen_type' and random.random() < 0.15:
                    new_type = random.choice(['dct', 'fft'])
                    g = (self._new_dct(g['name']) if new_type == 'dct'
                         else self._new_fft(g['name']))
                    break

                # For skip, the generator physically caps at 1/step pixels regardless of
                # capacity_ratio. Keep capacity honest so EA fitness reflects actual payload.
                if g['gen_type'] == 'lsb' and g.get('strategy') == 'skip' and g.get('step', 1) > 0:
                    g['capacity_ratio'] = min(g['capacity_ratio'], 1.0 / g['step'])

            elif gt == 'dct':
                field = random.choice(['coeff', 'strength', 'capacity', 'gen_type'])
                if field == 'coeff':
                    g['coeff_selection'] = random.choice(DCT_COEFF_MODES)
                elif field == 'strength':
                    g['strength'] = max(DCT_STRENGTH_RANGE[0], min(DCT_STRENGTH_RANGE[1],
                        g['strength'] + random.uniform(-1.5, 1.5)))
                elif field == 'capacity':
                    g['capacity_ratio'] = max(DCT_CAPACITY_RANGE[0], min(DCT_CAPACITY_RANGE[1],
                        g['capacity_ratio'] + random.uniform(-0.08, 0.08)))
                elif field == 'gen_type' and random.random() < 0.10:
                    g = self._new_fft(g['name'])
                    break

            elif gt == 'fft':
                field = random.choice(['band', 'strength', 'capacity', 'gen_type'])
                if field == 'band':
                    other_bands = [b for b in FFT_FREQ_BANDS if b != g['freq_band']]
                    g['freq_band'] = random.choice(other_bands)
                elif field == 'strength':
                    g['strength'] = max(FFT_STRENGTH_RANGE[0], min(FFT_STRENGTH_RANGE[1],
                        g['strength'] + random.uniform(-3.0, 3.0)))
                elif field == 'capacity':
                    g['capacity_ratio'] = max(FFT_CAPACITY_RANGE[0], min(FFT_CAPACITY_RANGE[1],
                        g['capacity_ratio'] + random.uniform(-0.05, 0.05)))
                elif field == 'gen_type' and random.random() < 0.10:
                    g = self._new_dct(g['name'])
                    break

            elif gt == 'adaptive':
                # Adaptive evolves only cost-model SHAPE — never payload (the
                # curriculum owns it) and never gen_type (stays in its niche).
                field = random.choice(['sigma_offset', 'cost_exponent', 'diagonal'])
                if field == 'sigma_offset':
                    g['sigma_offset'] = round(max(0.1, min(5.0,
                        g['sigma_offset'] + random.uniform(-0.5, 0.5))), 2)
                elif field == 'cost_exponent':
                    g['cost_exponent'] = round(max(0.5, min(2.0,
                        g['cost_exponent'] + random.uniform(-0.3, 0.3))), 2)
                elif field == 'diagonal':
                    g['use_diagonal'] = not g.get('use_diagonal', True)

        return g

    def crossover(self, g1: dict, g2: dict) -> dict:
        if get_niche(g1) == get_niche(g2) and random.random() < 0.7:
            g2 = self.mutate(g2)

        child = copy.deepcopy(g1)
        child['name'] = f"Cross_{self.generation}n{self._fresh_id()}"

        if g1['gen_type'] != g2['gen_type']:
            # Different gen_types — keep g1's parameters as-is. A cross-type
            # capacity copy would land outside the child's per-method range.
            return child

        gt = g1['gen_type']
        if gt == 'lsb':
            if random.random() < 0.5: child['step']     = g2['step']
            if random.random() < 0.5: child['strategy'] = g2['strategy']
        elif gt == 'dct':
            if random.random() < 0.5: child['coeff_selection'] = g2['coeff_selection']
            if random.random() < 0.5: child['strength']        = g2['strength']
        elif gt == 'fft':
            if random.random() < 0.5: child['freq_band'] = g2['freq_band']
            if random.random() < 0.5: child['strength']  = g2['strength']
        elif gt == 'adaptive':
            if random.random() < 0.5: child['sigma_offset']  = g2['sigma_offset']
            if random.random() < 0.5: child['cost_exponent'] = g2['cost_exponent']
            if random.random() < 0.5: child['use_diagonal']  = g2['use_diagonal']

        if gt != 'adaptive' and random.random() < 0.5:
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
            if g['gen_type'] != genome['gen_type']:
                continue
            cap_close = (abs(g.get('capacity_ratio', 0)
                             - genome.get('capacity_ratio', 0)) < 0.05)
            gt = genome['gen_type']
            if gt == 'adaptive':
                if (g.get('adaptive_mode') == genome.get('adaptive_mode')
                        and abs(g.get('sigma_offset', 0)
                                - genome.get('sigma_offset', 0)) < 0.3):
                    return True
            elif gt == 'lsb':
                if g.get('strategy') == genome.get('strategy') and cap_close:
                    return True
            elif gt == 'dct':
                if (g.get('coeff_selection') == genome.get('coeff_selection')
                        and abs(g.get('strength', 0) - genome.get('strength', 0)) < 0.5
                        and cap_close):
                    return True
            else:  # fft
                if (g.get('freq_band') == genome.get('freq_band')
                        and abs(g.get('strength', 0) - genome.get('strength', 0)) < 0.5
                        and cap_close):
                    return True
        return False

    def evolve(self) -> dict:
        """Run one generation: score, select, reproduce, inject. Returns best genome.

        Non-adaptive (LSB/DCT/FFT) and adaptive genomes evolve in SEPARATE pools.
        Adaptive often posts a high fool rate at low curriculum payloads; keeping
        the pools separate stops it from crowding LSB/DCT/FFT out of the elite
        and crossover breeding. Adaptive evolves cost-model SHAPE only — its
        payload is curriculum-controlled, never evolved.
        """
        self.generation += 1

        # Score (fitness = raw fool rate; capacity penalty retired).
        final_scores = {}
        for genome in self.population:
            data = self.stats[genome['name']]
            raw  = data['fooled'] / data['attempts'] if data['attempts'] > 0 else 0.0
            final_scores[genome['name']] = self._penalised_fitness(genome, raw)

        non_adaptive = [g for g in self.population if g['gen_type'] != 'adaptive']
        adaptive     = [g for g in self.population if g['gen_type'] == 'adaptive']

        sorted_na = sorted(non_adaptive,
                           key=lambda g: final_scores.get(g['name'], 0.0), reverse=True)
        sorted_ad = sorted(adaptive,
                           key=lambda g: final_scores.get(g['name'], 0.0), reverse=True)

        na_slots = POPULATION_SIZE - _ADAPTIVE_POP_SIZE
        non_adaptive_niches = [n for n in ALL_NICHES if not n.startswith('adaptive_')]

        # ── Non-adaptive breeding ────────────────────────────────────────────
        niche_survivors = []
        for niche in non_adaptive_niches:
            members = [g for g in sorted_na if get_niche(g) == niche]
            for g in members[:MIN_NICHE_SIZE]:
                if g not in niche_survivors:
                    niche_survivors.append(g)

        # Niche-type-diverse elite: best LSB, best DCT, best FFT.
        # Prevents a single niche (e.g. lsb_random) from occupying all three
        # parent slots and flooding the fill loop with homogeneous mutations.
        elite = []
        for niche_prefix in ('lsb', 'dct', 'fft'):
            best = next((g for g in sorted_na
                         if get_niche(g).startswith(niche_prefix)
                         and g not in elite), None)
            if best:
                elite.append(best)
        # Top-up to 3 if any type had zero genomes
        for g in sorted_na:
            if len(elite) >= 3:
                break
            if g not in elite:
                elite.append(g)

        # Print summary AFTER elite is built so logs show the actual breeding parents.
        self._print_evolution_summary(elite, sorted_ad, final_scores)

        new_na = list({id(g): g for g in elite + niche_survivors}.values())

        if sorted_na:
            # Intra-type crossover: pair each elite member with the next best
            # genome of the same gen_type. Same-type crossover actually mixes
            # parameters (coeff, strength, capacity, etc.); cross-type crossover
            # just returns a copy of g1, so this avoids wasted crossover slots.
            for parent_a in elite:
                gt = parent_a['gen_type']
                parent_b = next(
                    (g for g in sorted_na
                     if g['gen_type'] == gt and g['name'] != parent_a['name']),
                    None,
                )
                if parent_b:
                    child = self.crossover(parent_a, parent_b)
                else:
                    child = self.mutate(parent_a)
                if not self._is_duplicate(child, new_na):
                    new_na.append(child)
                else:
                    new_na.append(self.mutate(parent_a))

            # Inject the two most under-represented non-adaptive niches
            niche_counts_new = {n: sum(1 for g in new_na if get_niche(g) == n)
                                for n in non_adaptive_niches}
            underrepresented = sorted(non_adaptive_niches,
                                      key=lambda n: niche_counts_new.get(n, 0))
            for niche in underrepresented[:2]:
                if niche.startswith('lsb_'):
                    strategy = niche.split('_', 1)[1]
                    new_na.append(self._new_lsb(f"Explore_{niche}_{self.generation}", strategy))
                elif niche == 'dct':
                    new_na.append(self._new_dct(f"Explore_dct_{self.generation}"))
                elif niche.startswith('fft_'):
                    band = niche.split('_', 1)[1]
                    new_na.append(self._new_fft(f"Explore_{niche}_{self.generation}", band))

            # Fill remainder by mutating niche-diverse elite parents so fill
            # slots are spread across LSB/DCT/FFT, not all from the same niche.
            while len(new_na) < na_slots:
                parent_idx = min(random.randint(0, 2), len(elite) - 1)
                new_na.append(self.mutate(elite[parent_idx]))

        new_na = new_na[:na_slots]

        # ── Adaptive sub-population: shape evolution only ────────────────────
        # Keep the best MIN_NICHE_SIZE adaptive shapes; refresh the rest by
        # mutating the best (mutate() only touches sigma/exponent/diagonal).
        adaptive_pop = list(sorted_ad[:MIN_NICHE_SIZE])
        while len(adaptive_pop) < _ADAPTIVE_POP_SIZE and sorted_ad:
            adaptive_pop.append(self.mutate(sorted_ad[0]))

        self.population = new_na + adaptive_pop[:_ADAPTIVE_POP_SIZE]
        self.stats      = {g['name']: {'fooled': 0, 'attempts': 0} for g in self.population}
        return sorted_na[0] if sorted_na else self.population[0]

    def _print_evolution_summary(self, sorted_na: list, sorted_ad: list,
                                 final_scores: dict) -> None:
        print(f"\n[EVOLUTION] Generation {self.generation} — Top 3 non-adaptive:")
        for i in range(min(3, len(sorted_na))):
            g   = sorted_na[i]
            sc  = final_scores.get(g['name'], 0.0) * 100
            d   = self.stats[g['name']]
            raw = (d['fooled'] / d['attempts'] * 100) if d['attempts'] > 0 else 0.0
            gt  = g['gen_type']
            if gt == 'lsb':
                detail = (f"Strat={g['strategy']} Step={g['step']} "
                          f"Cap={g['capacity_ratio']:.2f}")
            elif gt == 'dct':
                detail = (f"Coeff={g['coeff_selection']} Str={g['strength']:.1f} "
                          f"Cap={g['capacity_ratio']:.2f}")
            else:  # fft
                detail = (f"Band={g['freq_band']} Str={g['strength']:.1f} "
                          f"Cap={g['capacity_ratio']:.2f}")
            print(f"  #{i+1}: {g['name']} — {sc:.1f}% (raw {raw:.1f}%) | "
                  f"Niche={get_niche(g)} | {detail}")

        if sorted_ad:
            g   = sorted_ad[0]
            d   = self.stats[g['name']]
            raw = (d['fooled'] / d['attempts'] * 100) if d['attempts'] > 0 else 0.0
            print(f"  Adaptive (shape-evolved, payload=curriculum) — best shape: "
                  f"Sig={g.get('sigma_offset', 1.0):.2f} "
                  f"Exp={g.get('cost_exponent', 1.0):.2f} "
                  f"Diag={g.get('use_diagonal', True)}  (raw fool {raw:.1f}%)")

        niche_counts = {n: sum(1 for g in self.population if get_niche(g) == n)
                        for n in ALL_NICHES}
        print(f"  Niches: {niche_counts}")

    # ── Sampling ──────────────────────────────────────────────────────────────

    def get_random_genome(self) -> dict:
        """
        Sample a NON-ADAPTIVE genome for the EA-driven batch slots, weighted by
        fitness and diversity-dampened by niche size.

        Adaptive is excluded here: it enters every batch only via the Layer-7
        floor (get_adaptive_genome) at a fixed share, with a curriculum-set
        payload — so it can never flood the EA-driven slots.
        """
        pool = [g for g in self.population if g['gen_type'] != 'adaptive']
        if not pool:
            pool = self.population

        if self.generation == 0 or random.random() < 0.3:
            return random.choice(pool)

        niche_counts = {}
        for g in pool:
            n = get_niche(g)
            niche_counts[n] = niche_counts.get(n, 0) + 1

        weights = []
        for g in pool:
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
        return random.choices(pool, weights=weights, k=1)[0]

    def get_low_capacity_genome(self) -> dict:
        """Return a non-adaptive genome with capacity_ratio below the low-cap threshold."""
        candidates = [g for g in self.population
                      if g['gen_type'] != 'adaptive' and is_low_capacity(g)]
        if candidates:
            return random.choice(candidates)

        # Fallback: a fresh low-payload genome.
        if random.random() < 0.5:
            g = self._new_lsb("tmp_lowcap_seq", 'sequential')
        else:
            g = self._new_fft("tmp_lowcap_fft_low", 'low')
        g['capacity_ratio'] = round(random.uniform(0.06, LOW_CAPACITY_THRESHOLD), 3)
        return g

    def get_lowstrength_fft_low_genome(self):
        """Return an fft_low genome for the Layer-6 slot. Falls back to a fresh genome."""
        candidates = [
            g for g in self.population
            if g.get('gen_type') == 'fft' and g.get('freq_band') == 'low'
        ]
        if candidates:
            return random.choice(candidates)
        return self._new_fft("tmp_fft_low", 'low')

    def get_lowstrength_dct_lowmid_genome(self):
        """Return a dct_low_mid genome for the Layer-5 slot. Falls back to a fresh genome."""
        candidates = [
            g for g in self.population
            if g.get('gen_type') == 'dct' and g.get('coeff_selection') == 'low_mid'
        ]
        if candidates:
            return random.choice(candidates)
        return self._new_dct("tmp_dct_low_mid", 'low_mid')

    def get_adaptive_genome(self, mode: str) -> dict:
        """Return an adaptive genome of the given mode.

        Half the time injects a randomly-sampled shape from the full parameter
        range so the model cannot overfit to the EA's converged sigma/exponent
        neighbourhood. The other half uses the EA's adversarially-evolved shapes.
        capacity_ratio is a placeholder — overridden by the curriculum at embed time.
        """
        if random.random() < 0.5:
            # Match validation parameter bounds exactly to prevent train/val drift
            return {
                'name':           f'rnd_adaptive_{mode}_g{self.generation}r{random.randint(1000, 9999)}',
                'gen_type':       'adaptive',
                'adaptive_mode':  mode,
                'sigma_offset':   round(random.uniform(0.5, 5.0), 2),
                'capacity_ratio': 0.40,
                'cost_exponent':  round(random.uniform(0.5, 2.0), 2),
                'use_diagonal':   random.choice([True, False]),
                'canonical':      True,
            }
        candidates = [
            g for g in self.population
            if g.get('gen_type') == 'adaptive' and g.get('adaptive_mode') == mode
        ]
        if candidates:
            return random.choice(candidates)
        return self._new_adaptive(f"tmp_adaptive_{mode}", mode)