"""Search utilities for the DNS feature-selection experiment.

Contains the validation-only fitness function, binary ACO and PSO classes,
a random-search baseline, and helpers that record search budgets and
(for Enhanced RLGWO) whether the controller's neural network was updated.
"""
import gc
import json
import random
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from tqdm import tqdm

N_FEATURES = 17
MIN_FEATURES, MAX_FEATURES, IDEAL, PENALTY = 8, 10, 9, 0.002


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def make_fitness(X_tr, y_tr, X_val, y_val, seed=42, n_jobs=1, n_trees=50):
    """Return (fitness_function, stats).

    The fitness is weighted F1 - FPR - PENALTY * |k - IDEAL| for 8 <= k <= 10,
    and 0 otherwise. Only validation data are used for scoring.
    stats counts every call and every Random Forest fit, so the real
    search budget can be reported.
    """
    stats = {"calls": 0, "rf_fits": 0, "seconds_in_fits": 0.0}

    def fitness(individual):
        stats["calls"] += 1
        mask = np.asarray(individual)
        if mask.shape != (N_FEATURES,) or not np.all(np.isin(mask, [0, 1])):
            raise ValueError("Expected a binary mask of length 17.")
        idx = np.flatnonzero(mask)
        k = len(idx)
        if not MIN_FEATURES <= k <= MAX_FEATURES:
            return (0.0,)
        start = time.perf_counter()
        clf = RandomForestClassifier(n_estimators=n_trees, random_state=seed, n_jobs=n_jobs)
        clf.fit(X_tr[:, idx], y_tr)
        pred = clf.predict(X_val[:, idx])
        stats["rf_fits"] += 1
        stats["seconds_in_fits"] += time.perf_counter() - start
        score = f1_score(y_val, pred, average="weighted", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        return (float(score - fpr - PENALTY * abs(k - IDEAL)),)

    return fitness, stats


class BinaryACO:
    """ACO feature selection adapted from Talabani et al. (2025).

    Reinforcement is proportional to the non-negative fitness, so better
    subsets receive more pheromone (an adaptation of the published equation).
    """

    def __init__(self, n_dimensions, population_size, generations,
                 evaporation_rate=0.5, alpha=1.0, beta=1.0, heuristic=None):
        self.n_dimensions = n_dimensions
        self.population_size = population_size
        self.generations = generations
        self.rho = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.heuristic = heuristic if heuristic is not None else np.ones(n_dimensions)

    def _construct_subset(self, pheromone):
        not_yet = list(range(self.n_dimensions))
        subset = []
        size = np.random.randint(1, self.n_dimensions + 1)
        for _ in range(size):
            if not not_yet:
                break
            weights = np.array([(pheromone[i] ** self.alpha) * (self.heuristic[i] ** self.beta)
                                for i in not_yet])
            weights = np.clip(weights, 1e-12, None)
            probs = weights / weights.sum()
            chosen = np.random.choice(not_yet, p=probs)
            subset.append(chosen)
            not_yet.remove(chosen)
        individual = np.zeros(self.n_dimensions, dtype=int)
        individual[subset] = 1
        return individual

    def run(self, evaluate_func):
        pheromone = np.ones(self.n_dimensions)
        best_ind, best_fit, history = None, -np.inf, []
        bar = tqdm(range(1, self.generations + 1), desc="Binary ACO")
        for _ in bar:
            population = np.array([self._construct_subset(pheromone) for _ in range(self.population_size)])
            fitnesses = np.array([evaluate_func(ind.tolist())[0] for ind in population])
            gen_best = int(np.argmax(fitnesses))
            if fitnesses[gen_best] > best_fit:
                best_fit = fitnesses[gen_best]
                best_ind = population[gen_best].copy()
            pheromone = (1 - self.rho) * pheromone
            for ind, fit in zip(population, fitnesses):
                pheromone += (self.rho * max(fit, 0.0)) * ind
            history.append(float(best_fit))
            bar.set_postfix(best=f"{best_fit:.4f}", k=int(best_ind.sum()))
        return best_ind.tolist(), float(best_fit), history


class BinaryPSO:
    """Binary PSO adapted from Tong and Wu (2014)."""

    def __init__(self, n_dimensions, population_size, generations,
                 w=0.9, c1=0.5, c2=0.3, v_max=6.0):
        self.n_dimensions = n_dimensions
        self.population_size = population_size
        self.generations = generations
        self.w, self.c1, self.c2, self.v_max = w, c1, c2, v_max

    @staticmethod
    def _sigmoid(v):
        return 1 / (1 + np.exp(-v))

    def run(self, evaluate_func):
        pos = np.random.randint(0, 2, (self.population_size, self.n_dimensions)).astype(float)
        vel = np.zeros((self.population_size, self.n_dimensions))
        fit = np.array([evaluate_func(p.astype(int).tolist())[0] for p in pos])
        pbest_pos, pbest_fit = pos.copy(), fit.copy()
        g = int(np.argmax(fit))
        gbest_pos, gbest_fit = pos[g].copy(), float(fit[g])
        history = [gbest_fit]
        bar = tqdm(range(1, self.generations + 1), desc="Binary PSO")
        for _ in bar:
            r1, r2 = np.random.rand(*pos.shape), np.random.rand(*pos.shape)
            vel = self.w * vel + self.c1 * r1 * (pbest_pos - pos) + self.c2 * r2 * (gbest_pos - pos)
            vel = np.clip(vel, -self.v_max, self.v_max)
            pos = (np.random.rand(*pos.shape) < self._sigmoid(vel)).astype(float)
            for i in range(self.population_size):
                if pos[i].sum() == 0:
                    pos[i, np.random.randint(self.n_dimensions)] = 1
            fit = np.array([evaluate_func(p.astype(int).tolist())[0] for p in pos])
            improved = fit > pbest_fit
            pbest_pos[improved] = pos[improved]
            pbest_fit[improved] = fit[improved]
            if fit.max() > gbest_fit:
                gbest_fit = float(fit.max())
                gbest_pos = pos[int(np.argmax(fit))].copy()
            history.append(gbest_fit)
            bar.set_postfix(best=f"{gbest_fit:.4f}", k=int(gbest_pos.sum()))
        return gbest_pos.astype(int).tolist(), gbest_fit, history


class RandomSearch:
    """Simple baseline: sample random admissible masks and keep the best.

    n_evals is the number of fitness calls. If k_fixed is given, every mask
    has exactly k_fixed features; otherwise k is drawn from 8..10.
    """

    def __init__(self, n_dimensions, n_evals, k_fixed=None):
        self.n_dimensions = n_dimensions
        self.n_evals = n_evals
        self.k_fixed = k_fixed

    def run(self, evaluate_func):
        best_ind, best_fit, history = None, -np.inf, []
        for _ in tqdm(range(self.n_evals), desc="Random search"):
            k = self.k_fixed or int(np.random.randint(MIN_FEATURES, MAX_FEATURES + 1))
            mask = np.zeros(self.n_dimensions, dtype=int)
            mask[np.random.choice(self.n_dimensions, size=k, replace=False)] = 1
            fit = evaluate_func(mask.tolist())[0]
            if fit > best_fit:
                best_fit, best_ind = fit, mask
            history.append(float(best_fit))
        return best_ind.tolist(), float(best_fit), history


def attach_update_counter(optimiser):
    """Count neural-network updates inside an Enhanced RLGWO-style object.

    Looks for a torch optimiser stored as `optimiser.optimizer` and wraps its
    step() method. Returns a dict that is updated during the run. If the
    attribute does not exist the dict says so instead of guessing.
    """
    counter = {"network_updates": 0, "tracked": False}
    inner = getattr(optimiser, "optimizer", None)
    if inner is not None and hasattr(inner, "step"):
        original = inner.step

        def counted(*args, **kwargs):
            counter["network_updates"] += 1
            return original(*args, **kwargs)

        inner.step = counted
        counter["tracked"] = True
    return counter


def describe_controller(optimiser):
    """Read common controller attributes after a run (whatever exists)."""
    info = {}
    for name in ["batch_size", "epsilon", "eps", "epsilon_end", "beta", "tau", "gamma", "afactor"]:
        if hasattr(optimiser, name):
            value = getattr(optimiser, name)
            if isinstance(value, (int, float, np.floating, np.integer)):
                info[name] = float(value)
    for name in ["replay_buffer", "memory"]:
        if hasattr(optimiser, name):
            try:
                info[f"{name}_length"] = len(getattr(optimiser, name))
            except TypeError:
                pass
    history = getattr(optimiser, "history", None)
    if isinstance(history, dict) and "actions" in history:
        actions = history["actions"]
        info["action_counts"] = {str(a): actions.count(a) for a in set(map(str, actions))} if actions else {}
    return info


def run_search(name, optimiser_factory, fitness, stats, seed, out_dir, extra=None):
    """Run one search with a fixed seed, save a JSON record, and return it."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name.lower().replace(' ', '_')}_seed{seed}_search.json"
    if path.exists():
        print("Already done, loading:", path.name)
        return json.loads(path.read_text())

    seed_everything(seed)
    stats.update({"calls": 0, "rf_fits": 0, "seconds_in_fits": 0.0})
    optimiser = optimiser_factory()
    counter = attach_update_counter(optimiser)
    start = time.perf_counter()
    solution, best_fit, history = optimiser.run(fitness)
    seconds = time.perf_counter() - start

    mask = np.asarray(solution, dtype=int)
    idx = np.flatnonzero(mask)
    if not MIN_FEATURES <= len(idx) <= MAX_FEATURES or best_fit <= 0:
        raise ValueError("Search did not return an admissible positive-fitness solution.")
    record = {
        "name": name, "seed": seed, "selected_indices": idx.tolist(),
        "best_validation_fitness": float(best_fit),
        "history": [float(v) for v in history],
        "search_seconds": seconds,
        "fitness_calls": stats["calls"], "rf_fits": stats["rf_fits"],
        "controller_updates": counter,
        "controller_info": describe_controller(optimiser),
        "search_evaluation_partition": "validation",
    }
    if extra:
        record.update(extra)
    path.write_text(json.dumps(record, indent=2))
    print({k: record[k] for k in ["name", "seed", "selected_indices", "best_validation_fitness",
                                  "fitness_calls", "rf_fits", "search_seconds"]})
    del optimiser
    gc.collect()
    return record
