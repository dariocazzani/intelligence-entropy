"""Tests for IPC evaluation module."""

from __future__ import annotations

from pathlib import Path

from ie.environments.ipc_evaluation import (
    _extract_block,
    _extract_predicates,
    count_unsatisfied_goals,
    evaluate_planners,
)

# --- PDDL parsing ---


class TestExtractBlock:
    """Tests for _extract_block."""

    def test_extracts_goal(self) -> None:
        text = "(:init (a)) (:goal (and (b) (c)))"
        block = _extract_block(text, "goal")
        assert block == "(:goal (and (b) (c)))"

    def test_extracts_init(self) -> None:
        text = "(:init (a) (b)) (:goal (c))"
        block = _extract_block(text, "init")
        assert block == "(:init (a) (b))"

    def test_missing_keyword(self) -> None:
        assert _extract_block("(:init (a))", "goal") == ""

    def test_nested_parens(self) -> None:
        text = "(:goal (and (on a b) (clear a)))"
        block = _extract_block(text, "goal")
        assert "(on a b)" in block
        assert "(clear a)" in block


class TestExtractPredicates:
    """Tests for _extract_predicates."""

    def test_simple_conjunction(self) -> None:
        block = "(:goal (and (on a b) (clear a)))"
        preds = _extract_predicates(block)
        assert len(preds) == 2
        assert "(on a b)" in preds
        assert "(clear a)" in preds

    def test_single_predicate(self) -> None:
        block = "(:goal (and (served p1)))"
        preds = _extract_predicates(block)
        assert preds == ["(served p1)"]

    def test_no_and_wrapper(self) -> None:
        block = "(:goal (served p1))"
        preds = _extract_predicates(block)
        assert preds == ["(served p1)"]

    def test_init_block(self) -> None:
        block = "(:init (arm-empty) (clear b1) (on b1 b2))"
        preds = _extract_predicates(block)
        assert len(preds) == 3


class TestCountUnsatisfiedGoals:
    """Tests for count_unsatisfied_goals."""

    def test_all_unsatisfied(self, tmp_path: Path) -> None:
        pddl = (
            "(define (problem test)\n"
            " (:domain test)\n"
            " (:objects a b)\n"
            " (:init (clear a))\n"
            " (:goal (and (on a b) (clear b))))\n"
        )
        f = tmp_path / "test.pddl"
        f.write_text(pddl)
        assert count_unsatisfied_goals(f) == 2

    def test_one_satisfied(self, tmp_path: Path) -> None:
        pddl = (
            "(define (problem test)\n"
            " (:domain test)\n"
            " (:objects a b)\n"
            " (:init (clear a) (on a b))\n"
            " (:goal (and (on a b) (clear b))))\n"
        )
        f = tmp_path / "test.pddl"
        f.write_text(pddl)
        assert count_unsatisfied_goals(f) == 1

    def test_all_satisfied(self, tmp_path: Path) -> None:
        pddl = (
            "(define (problem test)\n"
            " (:domain test)\n"
            " (:objects a)\n"
            " (:init (clear a))\n"
            " (:goal (and (clear a))))\n"
        )
        f = tmp_path / "test.pddl"
        f.write_text(pddl)
        assert count_unsatisfied_goals(f) == 0

    def test_real_blocksworld_format(self, tmp_path: Path) -> None:
        """Test with realistic blocksworld PDDL."""
        pddl = (
            "(define (problem blocksworld-01)\n"
            " (:domain blocksworld)\n"
            " (:objects b1 b2 b3 - object)\n"
            " (:init \n"
            "    (arm-empty)\n"
            "    (clear b1)\n"
            "    (on b1 b2)\n"
            "    (on-table b2))\n"
            " (:goal  (and \n"
            "    (clear b2)\n"
            "    (on b2 b1)\n"
            "    (on-table b1))))\n"
        )
        f = tmp_path / "test.pddl"
        f.write_text(pddl)
        # (clear b2) not in init, (on b2 b1) not in init, (on-table b1) not in init
        assert count_unsatisfied_goals(f) == 3


# --- I_cd computation ---


class TestEvaluatePlanners:
    """Tests for evaluate_planners."""

    def _make_entry(
        self,
        algo: str,
        domain: str,
        problem: str,
        coverage: int,
        cost: int,
    ) -> dict:
        return {
            "algorithm": algo,
            "domain": domain,
            "problem": problem,
            "coverage": coverage,
            "cost": cost,
        }

    def test_basic_evaluation(self) -> None:
        properties = {
            "A-d1-p1": self._make_entry("A", "d1", "p1", 1, 10),
            "A-d2-p1": self._make_entry("A", "d2", "p1", 1, 5),
        }
        goal_entropies = {("d1", "p1"): 3, ("d2", "p1"): 2}
        results = evaluate_planners(properties, goal_entropies)
        assert "A" in results
        res = results["A"]
        assert res["n_solved"] == 2
        assert res["n_domains"] == 2
        assert res["efficiency"] > 0
        assert res["context_diversity"] > 0
        assert res["i_cd"] > 0

    def test_unsolved_not_counted(self) -> None:
        properties = {
            "A-d1-p1": self._make_entry("A", "d1", "p1", 0, 0),
        }
        goal_entropies = {("d1", "p1"): 5}
        results = evaluate_planners(properties, goal_entropies)
        assert results["A"]["n_solved"] == 0
        assert results["A"]["i_cd"] == 0.0

    def test_single_domain_low_diversity(self) -> None:
        properties = {
            "A-d1-p1": self._make_entry("A", "d1", "p1", 1, 10),
            "A-d1-p2": self._make_entry("A", "d1", "p2", 1, 20),
        }
        goal_entropies = {("d1", "p1"): 3, ("d1", "p2"): 5}
        results = evaluate_planners(properties, goal_entropies)
        # Single domain -> H(C) = 0
        assert results["A"]["context_diversity"] == 0.0
        assert results["A"]["i_cd"] == 0.0

    def test_specialist_vs_generalist(self) -> None:
        """Specialist has higher eta but lower I_cd than generalist."""
        properties = {
            # Specialist: solves 2 problems in 1 domain very efficiently
            "Spec-d1-p1": self._make_entry("Spec", "d1", "p1", 1, 2),
            "Spec-d1-p2": self._make_entry("Spec", "d1", "p2", 1, 3),
            # Generalist: solves 1 problem each in 3 domains, less efficiently
            "Gen-d1-p1": self._make_entry("Gen", "d1", "p1", 1, 10),
            "Gen-d2-p1": self._make_entry("Gen", "d2", "p1", 1, 10),
            "Gen-d3-p1": self._make_entry("Gen", "d3", "p1", 1, 10),
        }
        goal_entropies = {
            ("d1", "p1"): 5,
            ("d1", "p2"): 5,
            ("d2", "p1"): 5,
            ("d3", "p1"): 5,
        }
        results = evaluate_planners(properties, goal_entropies)
        # Specialist: eta = 10/5 = 2.0, H(C) = 0 (single domain)
        assert results["Spec"]["efficiency"] > results["Gen"]["efficiency"]
        assert results["Spec"]["context_diversity"] == 0.0
        # Generalist: lower eta but positive H(C) -> positive I_cd
        assert results["Gen"]["i_cd"] > results["Spec"]["i_cd"]

    def test_eta_provably_positive(self) -> None:
        """All solved problems should yield positive eta."""
        properties = {
            "A-d1-p1": self._make_entry("A", "d1", "p1", 1, 100),
            "A-d2-p1": self._make_entry("A", "d2", "p1", 1, 200),
        }
        goal_entropies = {("d1", "p1"): 1, ("d2", "p1"): 1}
        results = evaluate_planners(properties, goal_entropies)
        assert results["A"]["efficiency"] > 0

    def test_multiple_planners(self) -> None:
        properties = {
            "A-d1-p1": self._make_entry("A", "d1", "p1", 1, 10),
            "B-d1-p1": self._make_entry("B", "d1", "p1", 1, 20),
        }
        goal_entropies = {("d1", "p1"): 5}
        results = evaluate_planners(properties, goal_entropies)
        assert len(results) == 2
        # A has shorter plan -> higher efficiency
        assert results["A"]["efficiency"] > results["B"]["efficiency"]
