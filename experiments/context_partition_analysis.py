"""Context partition analysis via Agglomerative Information Bottleneck.

Addresses the unanimous reviewer concern about context partition dependence
by showing that IB discovers domain structure from planner performance data alone.

Part 1: aIB on the 900x8 binary coverage matrix (pre-merged to ~35 unique profiles)
Part 2: AMI / ARI / V-measure comparison at various cluster counts
Part 3: I_cd ranking stability under different partitions
Part 4: IB curve (information retained vs number of clusters)

Usage:
    uv run python experiments/context_partition_analysis.py
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

PROJECT_ROOT = Path(__file__).parent.parent
IPC_PROPERTIES_PATH = PROJECT_ROOT / "data" / "ipc" / "properties.json"
RESULTS_DIR = Path(__file__).parent / "results"

CHECKPOINTS = [2, 3, 5, 10, 15, 20, 30]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_coverage_matrix():
    """Build problem x planner binary coverage matrix from IPC data.

    Returns:
        coverage: (N, P) binary matrix
        problem_ids: list of (domain, problem) tuples
        planner_names: list of planner name strings
        domain_labels: array of domain indices per problem
        domain_names: list of domain name strings
        properties: raw properties dict
    """
    with open(IPC_PROPERTIES_PATH) as f:
        properties = json.load(f)

    planners = sorted(set(e["algorithm"] for e in properties.values()))
    problems = sorted(set((e["domain"], e["problem"]) for e in properties.values()))

    planner_idx = {name: i for i, name in enumerate(planners)}
    problem_idx = {pid: i for i, pid in enumerate(problems)}

    n_problems, n_planners = len(problems), len(planners)
    coverage = np.zeros((n_problems, n_planners), dtype=np.float64)

    for entry in properties.values():
        pid = (entry["domain"], entry["problem"])
        pi = problem_idx[pid]
        ai = planner_idx[entry["algorithm"]]
        coverage[pi, ai] = entry["coverage"]

    domains = sorted(set(d for d, _ in problems))
    domain_map = {d: i for i, d in enumerate(domains)}
    domain_labels = np.array([domain_map[d] for d, _ in problems])

    return coverage, problems, planners, domain_labels, domains, properties


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------


def _h(p: np.ndarray) -> float:
    """Shannon entropy H(p) in nats."""
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _js(p: np.ndarray, q: np.ndarray, w1: float, w2: float) -> float:
    """Weighted Jensen-Shannon divergence JS_{w1,w2}(p, q)."""
    m = w1 * p + w2 * q
    return _h(m) - w1 * _h(p) - w2 * _h(q)


# ---------------------------------------------------------------------------
# Clustering metrics (no sklearn dependency)
# ---------------------------------------------------------------------------


def _contingency(labels_a: np.ndarray, labels_b: np.ndarray):
    """Build contingency matrix and marginals."""
    n = len(labels_a)
    pairs: dict[tuple[int, int], int] = defaultdict(int)
    for i in range(n):
        pairs[(int(labels_a[i]), int(labels_b[i]))] += 1

    a_vals = sorted(set(int(x) for x in labels_a))
    b_vals = sorted(set(int(x) for x in labels_b))
    a_map = {v: i for i, v in enumerate(a_vals)}
    b_map = {v: i for i, v in enumerate(b_vals)}

    ct = np.zeros((len(a_vals), len(b_vals)), dtype=np.float64)
    for (a, b), count in pairs.items():
        ct[a_map[a], b_map[b]] = count

    return ct, n


def _log_comb(n: int, k: int) -> float:
    """Log of binomial coefficient using lgamma."""
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def adjusted_mutual_info(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """AMI with arithmetic mean normalization (matches sklearn default)."""
    ct, n = _contingency(labels_a, labels_b)
    if n == 0:
        return 0.0

    a_sum = ct.sum(axis=1)
    b_sum = ct.sum(axis=0)

    h_a = _h(a_sum / n)
    h_b = _h(b_sum / n)

    # MI(A, B)
    mi = 0.0
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            if ct[i, j] > 0:
                mi += (ct[i, j] / n) * math.log(n * ct[i, j] / (a_sum[i] * b_sum[j]))

    # E[MI] under permutation model (Vinh et al. 2010)
    emi = 0.0
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ai = int(a_sum[i])
            bj = int(b_sum[j])
            lo = max(1, ai + bj - n)
            hi = min(ai, bj)
            for nij in range(lo, hi + 1):
                log_prob = _log_comb(ai, nij) + _log_comb(n - ai, bj - nij) - _log_comb(n, bj)
                if log_prob > -500:
                    prob = math.exp(log_prob)
                    if nij > 0:
                        emi += (nij / n) * math.log(n * nij / (ai * bj)) * prob

    denom = 0.5 * (h_a + h_b) - emi
    if abs(denom) < 1e-15:
        return 0.0 if abs(mi - emi) < 1e-15 else 1.0
    return (mi - emi) / denom


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """ARI from contingency table."""
    ct, n = _contingency(labels_a, labels_b)
    a_sum = ct.sum(axis=1)
    b_sum = ct.sum(axis=0)

    def comb2(x):
        return x * (x - 1) / 2

    sum_cij2 = sum(comb2(ct[i, j]) for i in range(ct.shape[0]) for j in range(ct.shape[1]))
    sum_ai2 = sum(comb2(a) for a in a_sum)
    sum_bj2 = sum(comb2(b) for b in b_sum)
    comb_n = comb2(n)

    if comb_n == 0:
        return 0.0

    expected = sum_ai2 * sum_bj2 / comb_n
    max_idx = 0.5 * (sum_ai2 + sum_bj2)
    denom = max_idx - expected
    if abs(denom) < 1e-15:
        return 0.0 if abs(sum_cij2 - expected) < 1e-15 else 1.0
    return (sum_cij2 - expected) / denom


def homogeneity_completeness_v(labels_true: np.ndarray, labels_pred: np.ndarray):
    """V-measure decomposition."""
    ct, n = _contingency(labels_true, labels_pred)
    if n == 0:
        return 0.0, 0.0, 0.0

    h_true = _h(ct.sum(axis=1) / n)
    h_pred = _h(ct.sum(axis=0) / n)

    h_true_given_pred = 0.0
    for j in range(ct.shape[1]):
        col = ct[:, j]
        col_sum = col.sum()
        if col_sum > 0:
            h_true_given_pred += (col_sum / n) * _h(col / col_sum)

    h_pred_given_true = 0.0
    for i in range(ct.shape[0]):
        row = ct[i, :]
        row_sum = row.sum()
        if row_sum > 0:
            h_pred_given_true += (row_sum / n) * _h(row / row_sum)

    homogeneity = 1.0 - h_true_given_pred / h_true if h_true > 0 else 1.0
    completeness = 1.0 - h_pred_given_true / h_pred if h_pred > 0 else 1.0

    if homogeneity + completeness == 0:
        v_measure = 0.0
    else:
        v_measure = 2 * homogeneity * completeness / (homogeneity + completeness)

    return homogeneity, completeness, v_measure


# ---------------------------------------------------------------------------
# Agglomerative Information Bottleneck
# ---------------------------------------------------------------------------


def _premerge_identical(coverage: np.ndarray):
    """Group problems with identical coverage profiles.

    Returns:
        groups: list of lists (each group = list of original problem indices)
        group_profiles: (G, P) array of unique coverage rows
        group_weights: (G,) array, group_weights[g] = |group_g| / N
    """
    profile_map: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for i in range(coverage.shape[0]):
        key = tuple(coverage[i].tolist())
        profile_map[key].append(i)

    keys = sorted(profile_map.keys())
    groups = [profile_map[k] for k in keys]
    group_profiles = np.array(keys)
    n_total = coverage.shape[0]
    group_weights = np.array([len(g) / n_total for g in groups])

    return groups, group_profiles, group_weights


def agglomerative_ib(coverage: np.ndarray):
    """Run agglomerative IB on the binary coverage matrix.

    Pre-merges identical profiles for efficiency (900 problems -> ~35 groups),
    then runs the standard aIB merge loop on the reduced set.

    Returns:
        merge_history: list of (i, j, cost, n_clusters_before)
        cluster_assignments: dict k -> labels array (length N_original)
        ib_curve: list of (k, I(T;Y)) pairs
    """
    n_orig, n_plan = coverage.shape

    # Pre-merge identical profiles
    groups, group_profiles, group_weights = _premerge_identical(coverage)
    n_groups = len(groups)
    print(f"  Pre-merged {n_orig} problems into {n_groups} unique coverage profiles")

    # p(y|t) for each group
    row_sums = group_profiles.sum(axis=1, keepdims=True)
    has_solution = row_sums.ravel() > 0
    py_given_t = group_profiles.copy()
    py_given_t[has_solution] /= row_sums[has_solution]
    py_given_t[~has_solution] = 1.0 / n_plan

    pt = group_weights.copy()
    members = {i: [i] for i in range(n_groups)}  # cluster -> member group indices
    active = set(range(n_groups))

    # Marginal p(y)
    py = np.zeros(n_plan)
    for i in range(n_groups):
        py += pt[i] * py_given_t[i]

    # Initial I(T; Y)
    i_ty_init = 0.0
    for i in range(n_groups):
        for y in range(n_plan):
            if py_given_t[i, y] > 0 and py[y] > 0:
                i_ty_init += pt[i] * py_given_t[i, y] * math.log(py_given_t[i, y] / py[y])

    # Pairwise merge costs
    cost = np.full((n_groups, n_groups), np.inf)
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            w1 = pt[i] / (pt[i] + pt[j])
            w2 = pt[j] / (pt[i] + pt[j])
            cost[i, j] = (pt[i] + pt[j]) * _js(py_given_t[i], py_given_t[j], w1, w2)

    merge_history = []
    ib_curve = [(n_groups, i_ty_init)]
    cumulative_cost = 0.0
    checkpoint_set = set(CHECKPOINTS)
    cluster_assignments = {}

    def _snapshot_labels():
        """Map current clusters back to original problem indices."""
        labels = np.zeros(n_orig, dtype=int)
        for cid_num, cid in enumerate(sorted(active)):
            for gid in members[cid]:
                for pid in groups[gid]:
                    labels[pid] = cid_num
        return labels

    print(f"  Running {n_groups - 1} merge steps...")
    for step in range(n_groups - 1):
        n_clusters = n_groups - step

        if n_clusters in checkpoint_set:
            cluster_assignments[n_clusters] = _snapshot_labels()

        flat_idx = np.argmin(cost)
        i, j = np.unravel_index(flat_idx, cost.shape)
        merge_cost_val = float(cost[i, j])

        if merge_cost_val == np.inf:
            break

        cumulative_cost += merge_cost_val
        merge_history.append((int(i), int(j), merge_cost_val, n_clusters))
        ib_curve.append((n_clusters - 1, i_ty_init - cumulative_cost))

        # Merge j into i
        w1 = pt[i] / (pt[i] + pt[j])
        w2 = pt[j] / (pt[i] + pt[j])
        py_given_t[i] = w1 * py_given_t[i] + w2 * py_given_t[j]
        pt[i] = pt[i] + pt[j]
        members[i].extend(members[j])
        del members[j]

        active.remove(j)
        cost[j, :] = np.inf
        cost[:, j] = np.inf

        for k in active:
            if k == i:
                continue
            ii, jj = min(i, k), max(i, k)
            w1 = pt[ii] / (pt[ii] + pt[jj])
            w2 = pt[jj] / (pt[ii] + pt[jj])
            cost[ii, jj] = (pt[ii] + pt[jj]) * _js(py_given_t[ii], py_given_t[jj], w1, w2)

    cluster_assignments[1] = np.zeros(n_orig, dtype=int)

    return merge_history, cluster_assignments, ib_curve


# ---------------------------------------------------------------------------
# I_cd under arbitrary partition
# ---------------------------------------------------------------------------


def compute_icd_under_partition(
    properties: dict,
    partition_labels: np.ndarray,
    problem_ids: list[tuple[str, str]],
    planner_names: list[str],
    goal_entropies: dict | None = None,
) -> dict[str, float]:
    """Compute I_cd for each planner using the given context partition.

    eta = total(-delta_S) / total(W) is partition-independent.
    H(C) = entropy over the distribution of solved problems across clusters.
    """
    problem_to_label = {pid: int(partition_labels[i]) for i, pid in enumerate(problem_ids)}

    results = {}
    for planner in planner_names:
        total_delta_s = 0.0
        total_work = 0.0
        context_labels = []

        for entry in properties.values():
            if entry["algorithm"] != planner or entry["coverage"] != 1:
                continue
            pid = (entry["domain"], entry["problem"])
            if pid not in problem_to_label:
                continue

            if goal_entropies is not None:
                s_init = goal_entropies.get(pid, 0)
            else:
                s_init = 1  # unit proxy; rankings driven by H(C) differences

            if s_init == 0:
                continue
            plan_cost = entry["cost"]
            if plan_cost <= 0:
                continue

            total_delta_s += -s_init
            total_work += plan_cost
            context_labels.append(problem_to_label[pid])

        if total_work == 0 or len(context_labels) == 0:
            results[planner] = 0.0
            continue

        eta = -total_delta_s / total_work

        counts = Counter(context_labels)
        total = sum(counts.values())
        hc = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                hc -= p * math.log(p)

        results[planner] = eta * hc

    return results


def _try_load_goal_entropies():
    """Try to load goal entropies from the PDDL benchmarks."""
    try:
        from ie.environments.ipc_evaluation import IPC_BENCHMARKS_DIR, compute_goal_entropies

        if IPC_BENCHMARKS_DIR.exists():
            return compute_goal_entropies()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("Context Partition Analysis via Agglomerative Information Bottleneck")
    print("=" * 70)

    # 1. Load data
    print("\n1. Loading IPC 2023 data...")
    coverage, problem_ids, planner_names, domain_labels, domain_names, properties = (
        load_coverage_matrix()
    )
    n_prob, n_plan = coverage.shape
    n_solved_any = int((coverage.sum(axis=1) > 0).sum())
    n_unsolved_all = n_prob - n_solved_any
    print(f"   {n_prob} problems x {n_plan} planners")
    print(f"   {n_solved_any} solved by >= 1 planner, {n_unsolved_all} unsolved by all")
    print(f"   {len(domain_names)} domains: {domain_names}")

    # 2. Run aIB
    print("\n2. Running Agglomerative Information Bottleneck...")
    merge_history, cluster_assignments, ib_curve = agglomerative_ib(coverage)
    print(f"   Completed {len(merge_history)} merges")

    # 3. IB curve
    print("\n3. Information curve I(T; Y) vs number of clusters:")
    i_ty_max = ib_curve[0][1]
    display_ks = {1, 2, 3, 5, 10, 15, 20, 30, max(k for k, _ in ib_curve)}
    print(f"   {'k':>5}  {'I(T;Y)':>10}  {'% retained':>12}")
    for k, i_ty in ib_curve:
        if k in display_ks:
            pct = 100.0 * i_ty / i_ty_max if i_ty_max > 0 else 0
            print(f"   {k:>5}  {i_ty:>10.4f}  {pct:>11.1f}%")

    # 4. Clustering comparison
    print("\n4. Clustering comparison (IB clusters vs domain labels):")
    print(f"   {'k':>5}  {'AMI':>8}  {'ARI':>8}  {'V-meas':>8}  {'Homog':>8}  {'Compl':>8}")
    for k in sorted(cluster_assignments.keys()):
        if k <= 1:
            continue
        labels = cluster_assignments[k]
        ami_val = adjusted_mutual_info(domain_labels, labels)
        ari_val = adjusted_rand_index(domain_labels, labels)
        hom, com, vm = homogeneity_completeness_v(domain_labels, labels)
        print(f"   {k:>5}  {ami_val:>8.3f}  {ari_val:>8.3f}  {vm:>8.3f}  {hom:>8.3f}  {com:>8.3f}")

    # Detailed cluster composition at k=10
    if 10 in cluster_assignments:
        print("\n   IB@k=10 cluster composition:")
        ib10 = cluster_assignments[10]
        for c in range(10):
            mask = ib10 == c
            total_in_cluster = int(mask.sum())
            if total_in_cluster == 0:
                continue
            domain_counts = Counter(domain_labels[mask])
            desc = ", ".join(f"{domain_names[d]}:{n}" for d, n in domain_counts.most_common(3))
            print(f"     Cluster {c:>2} ({total_in_cluster:>3} problems): {desc}")

    # 5. I_cd ranking stability
    print("\n5. I_cd ranking stability across partitions:")
    goal_entropies = _try_load_goal_entropies()
    if goal_entropies is None:
        print("   (PDDL benchmarks not found; using unit S(E) proxy for eta)")

    domain_icd = compute_icd_under_partition(
        properties, domain_labels, problem_ids, planner_names, goal_entropies
    )
    domain_ranking = sorted(planner_names, key=lambda p: -domain_icd[p])

    print("\n   Domain partition (k=10, canonical):")
    for rank, name in enumerate(domain_ranking, 1):
        print(f"     {rank}. {name:<15} I_cd = {domain_icd[name]:.4f}")

    ranking_taus = {}
    for k in sorted(cluster_assignments.keys()):
        if k <= 1:
            continue
        ib_icd = compute_icd_under_partition(
            properties, cluster_assignments[k], problem_ids, planner_names, goal_entropies
        )
        ib_ranking = sorted(planner_names, key=lambda p: -ib_icd[p])

        domain_ranks = [domain_ranking.index(p) for p in planner_names]
        ib_ranks = [ib_ranking.index(p) for p in planner_names]
        tau, p_val = kendalltau(domain_ranks, ib_ranks)
        ranking_taus[k] = (tau, p_val)

        print(f"\n   IB partition (k={k}): Kendall tau = {tau:.3f} (p={p_val:.4f})")
        for rank, name in enumerate(ib_ranking, 1):
            print(f"     {rank}. {name:<15} I_cd = {ib_icd[name]:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if 10 in cluster_assignments:
        ami_10 = adjusted_mutual_info(domain_labels, cluster_assignments[10])
        ari_10 = adjusted_rand_index(domain_labels, cluster_assignments[10])
        print(f"  IB@k=10 vs domains: AMI={ami_10:.3f}, ARI={ari_10:.3f}")

    if ranking_taus:
        for k in sorted(ranking_taus.keys()):
            tau, pval = ranking_taus[k]
            print(f"  Kendall tau at k={k}: {tau:.3f} (p={pval:.4f})")

    if i_ty_max > 0:
        drops = []
        for i in range(1, len(ib_curve)):
            k_prev, i_prev = ib_curve[i - 1]
            k_curr, i_curr = ib_curve[i]
            drops.append((i_prev - i_curr, k_prev))
        drops.sort(reverse=True)
        print(f"  Largest info drops at k = {[d[1] for d in drops[:5]]}")

    # 6. Save results
    output = {
        "n_problems": n_prob,
        "n_planners": n_plan,
        "n_unique_profiles": len(set(tuple(r) for r in coverage)),
        "ib_curve": [(k, float(v)) for k, v in ib_curve],
        "domain_icd": domain_icd,
        "ranking_taus": {str(k): {"tau": t, "p_value": p} for k, (t, p) in ranking_taus.items()},
    }

    clustering_metrics = {}
    for k in sorted(cluster_assignments.keys()):
        if k <= 1:
            continue
        labels = cluster_assignments[k]
        hom, com, vm = homogeneity_completeness_v(domain_labels, labels)
        clustering_metrics[str(k)] = {
            "ami": adjusted_mutual_info(domain_labels, labels),
            "ari": adjusted_rand_index(domain_labels, labels),
            "v_measure": vm,
            "homogeneity": hom,
            "completeness": com,
        }
    output["clustering_metrics"] = clustering_metrics

    for k in sorted(cluster_assignments.keys()):
        if k <= 1:
            continue
        ib_icd = compute_icd_under_partition(
            properties, cluster_assignments[k], problem_ids, planner_names, goal_entropies
        )
        output[f"ib_k{k}_icd"] = ib_icd

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "context_partition_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
