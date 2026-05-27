"""IPC 2023 Learning Track evaluation for I_cd.

Computes I_cd from archival competition data: PDDL problem files provide
S(E) = unsatisfied goal predicates, plan cost provides W, and domain labels
provide context diversity H(C).

S(E) is provably non-negative and S(goal) = 0 by construction for any
solved planning problem, guaranteeing eta > 0 for all successful planners.
"""

from __future__ import annotations

import json
from pathlib import Path

from ie.measure import context_diversity, efficiency

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
IPC_DATA_DIR = PROJECT_ROOT / "data" / "ipc"
IPC_BENCHMARKS_DIR = IPC_DATA_DIR / "benchmarks"
IPC_PROPERTIES_PATH = IPC_DATA_DIR / "properties.json"


# --- PDDL parsing ---


def _extract_block(text: str, keyword: str) -> str:
    """Extract a PDDL block by matching balanced parentheses."""
    idx = text.find(f"(:{keyword}")
    if idx == -1:
        return ""
    depth = 0
    for i in range(idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[idx : i + 1]
    return ""


def _extract_predicates(block: str) -> list[str]:
    """Extract all atomic predicates from a PDDL block."""
    preds: list[str] = []
    i = 0
    while i < len(block):
        if block[i] == "(":
            depth = 1
            j = i + 1
            while j < len(block) and depth > 0:
                if block[j] == "(":
                    depth += 1
                elif block[j] == ")":
                    depth -= 1
                j += 1
            sub = block[i:j].strip()
            inner = sub[1:-1].strip()
            if inner.startswith(":goal") or inner.startswith(":init"):
                preds.extend(_extract_predicates(inner))
            elif inner.startswith("and ") or inner.startswith("and\n"):
                preds.extend(_extract_predicates(inner[3:]))
            elif "(" in inner:
                preds.extend(_extract_predicates(inner))
            else:
                preds.append(sub)
            i = j
        else:
            i += 1
    return preds


def _normalize(s: str) -> str:
    """Normalize whitespace for predicate comparison."""
    return " ".join(s.split())


def count_unsatisfied_goals(pddl_path: Path) -> int:
    """Count goal predicates not satisfied in the initial state.

    This is S(E_initial) for planning: how many goal conditions remain
    to be achieved. S(E_final) = 0 for any solved problem.
    """
    text = pddl_path.read_text()
    goal_block = _extract_block(text, "goal")
    init_block = _extract_block(text, "init")
    goal_preds = _extract_predicates(goal_block)
    init_preds = {_normalize(p) for p in _extract_predicates(init_block)}
    return sum(1 for p in goal_preds if _normalize(p) not in init_preds)


def resolve_problem_path(
    domain: str,
    problem_name: str,
    benchmarks_dir: Path = IPC_BENCHMARKS_DIR,
) -> Path | None:
    """Map IPC problem name to PDDL file path.

    Properties file uses 'easy-p01.pddl' format; filesystem uses
    'testing/easy/p01.pddl'.
    """
    parts = problem_name.rsplit("-", 1)
    if len(parts) != 2:
        return None
    difficulty, filename = parts
    path = benchmarks_dir / domain / "testing" / difficulty / filename
    return path if path.exists() else None


# --- Data loading ---


def load_properties(path: Path = IPC_PROPERTIES_PATH) -> dict[str, dict]:
    """Load the IPC properties JSON file."""
    with open(path) as f:
        result: dict[str, dict] = json.load(f)
    return result


def compute_goal_entropies(
    benchmarks_dir: Path = IPC_BENCHMARKS_DIR,
) -> dict[tuple[str, str], int]:
    """Compute S(E) = unsatisfied goal predicates for all problems.

    Returns:
        Mapping of (domain, problem_name) -> unsatisfied count.
    """
    entropies: dict[tuple[str, str], int] = {}
    for domain_dir in sorted(benchmarks_dir.iterdir()):
        testing_dir = domain_dir / "testing"
        if not testing_dir.is_dir():
            continue
        domain = domain_dir.name
        for diff_dir in sorted(testing_dir.iterdir()):
            if not diff_dir.is_dir():
                continue
            for pddl_file in sorted(diff_dir.glob("*.pddl")):
                problem_name = f"{diff_dir.name}-{pddl_file.name}"
                entropies[(domain, problem_name)] = count_unsatisfied_goals(pddl_file)
    return entropies


# --- I_cd computation ---


def evaluate_planners(
    properties: dict[str, dict],
    goal_entropies: dict[tuple[str, str], int],
) -> dict[str, dict]:
    """Compute I_cd for each planner in the properties data.

    S(E) = unsatisfied goal predicates (S_final = 0 for solved problems).
    W = plan cost (= plan length for unit-cost domains).
    Context = planning domain.

    Returns:
        Mapping of planner_name -> result dict with keys:
        efficiency, context_diversity, i_cd, n_solved, n_total, n_domains.
    """
    # Group entries by planner
    planner_entries: dict[str, list[dict]] = {}
    for entry in properties.values():
        algo = entry["algorithm"]
        planner_entries.setdefault(algo, []).append(entry)

    results: dict[str, dict] = {}
    for planner, entries in sorted(planner_entries.items()):
        total_delta_s = 0.0
        total_work = 0.0
        context_labels: list[int] = []
        domain_set: set[str] = set()
        n_solved = 0

        for entry in entries:
            if entry["coverage"] != 1:
                continue

            domain = entry["domain"]
            problem = entry["problem"]
            s_init = goal_entropies.get((domain, problem), 0)
            if s_init == 0:
                continue

            plan_cost = entry["cost"]
            if plan_cost <= 0:
                continue

            n_solved += 1
            delta_s = 0 - s_init  # S_final = 0, so delta = -S_init
            total_delta_s += delta_s
            total_work += plan_cost
            domain_set.add(domain)
            context_labels.append(hash(domain) % (2**31))

        eta = efficiency(total_delta_s, total_work)
        h_c = context_diversity(context_labels) if context_labels else 0.0
        score = eta * h_c

        results[planner] = {
            "i_cd": score,
            "efficiency": eta,
            "context_diversity": h_c,
            "n_solved": n_solved,
            "n_total": len(entries),
            "n_domains": len(domain_set),
        }

    return results
