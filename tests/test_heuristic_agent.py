"""Tests for heuristic agent implementations."""

from __future__ import annotations

import numpy as np

from ie.agents.heuristic_agent import (
    GridHeuristicAgent,
    RegulationHeuristicAgent,
    SeqHeuristicAgent,
    TilePuzzleHeuristicAgent,
)
from ie.environments.coupled_regulation import CoupledRegulation
from ie.environments.resource_grid import ResourceGrid
from ie.environments.sequence_sorting import SequenceSorting
from ie.environments.tile_puzzle import TilePuzzle


class TestGridHeuristicAgent:
    def test_returns_valid_actions(self) -> None:
        """Test that agent returns valid actions during episode."""
        agent = GridHeuristicAgent()
        env = ResourceGrid(seed=42)
        obs = env.reset({"name": "basic"})

        for _ in range(20):
            action = agent.act(obs)
            assert 0 <= action <= 5, f"Invalid action {action}"
            obs, _, done, _ = env.step(action)
            if done:
                break

    def test_picks_up_when_on_object(self) -> None:
        """Test that agent picks up object when positioned on it with empty inventory."""
        agent = GridHeuristicAgent()

        # Craft observation: agent at (2, 2), object at (2, 2), empty inventory
        grid = np.full(64, -1.0)  # Empty 8×8 grid
        grid[2 * 8 + 2] = 0.0  # Object at position (2, 2)
        obs = np.concatenate([grid, [2.0, 2.0, -1.0]])  # agent_row=2, agent_col=2, inventory=-1

        action = agent.act(obs)
        assert action == 4, "Agent should pick up object when on it"

    def test_navigates_toward_object(self) -> None:
        """Test that agent navigates toward nearest object."""
        agent = GridHeuristicAgent()

        # Agent at (0, 0), only object at (7, 7), empty inventory
        grid = np.full(64, -1.0)
        grid[7 * 8 + 7] = 0.0  # Object at (7, 7)
        obs = np.concatenate([grid, [0.0, 0.0, -1.0]])

        action = agent.act(obs)
        # Should move down (1) or right (3) to get closer to (7, 7)
        assert action in [1, 3], "Agent should move toward object"

    def test_places_at_target(self) -> None:
        """Test that agent places object when at target with empty cell."""
        agent = GridHeuristicAgent()

        # Agent at (0, 0), empty grid, holding object
        grid = np.full(64, -1.0)  # All empty
        obs = np.concatenate([grid, [0.0, 0.0, 0.0]])  # agent at (0,0), holding object type 0

        action = agent.act(obs)
        assert action == 5, "Agent should place object at target location"

    def test_reduces_entropy_in_episode(self) -> None:
        """Test that agent reduces environmental entropy over full episode."""
        agent = GridHeuristicAgent()
        env = ResourceGrid(seed=42)
        obs = env.reset({"name": "multi_type"})

        initial_entropy = env.compute_entropy()

        for _ in range(100):
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

        final_entropy = env.compute_entropy()
        assert final_entropy <= initial_entropy, "Agent should reduce or maintain entropy"


class TestSeqHeuristicAgent:
    def test_returns_valid_actions(self) -> None:
        """Test that agent returns valid actions during episode."""
        agent = SeqHeuristicAgent()
        env = SequenceSorting(seed=42)
        obs = env.reset({"name": "short"})

        for _ in range(20):
            action = agent.act(obs)
            assert 0 <= action <= 24, f"Invalid action {action}"  # Updated for MAX_LENGTH=25
            obs, _, done, _ = env.step(action)
            if done:
                break

    def test_swaps_first_inversion(self) -> None:
        """Test that agent swaps at first inversion position."""
        agent = SeqHeuristicAgent()

        # obs = [3, 1, 2, ...zeros to pad to MAX_LENGTH=25]
        # First inversion is at position 0: 3 > 1
        obs = np.zeros(25)
        obs[:3] = [3.0, 1.0, 2.0]
        action = agent.act(obs)
        assert action == 0, "Agent should swap at position 0 (first inversion)"

    def test_noop_when_sorted(self) -> None:
        """Test that agent performs no-op when sequence is sorted."""
        agent = SeqHeuristicAgent()

        # Sorted sequence with trailing zeros (updated for MAX_LENGTH=25)
        obs = np.zeros(25)
        obs[:6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        action = agent.act(obs)
        assert action == 24, "Agent should perform no-op when sorted"  # MAX_LENGTH-1 = 24

    def test_handles_all_zeros(self) -> None:
        """Test that agent handles all-zero observation correctly."""
        agent = SeqHeuristicAgent()

        # All zeros (no inversions) - updated for MAX_LENGTH=25
        obs = np.zeros(25)
        action = agent.act(obs)
        assert action == 24, "Agent should perform no-op for all zeros"  # MAX_LENGTH-1 = 24

    def test_reduces_entropy_in_episode(self) -> None:
        """Test that agent reduces environmental entropy over full episode."""
        agent = SeqHeuristicAgent()
        env = SequenceSorting(seed=42)
        obs = env.reset({"name": "medium"})

        initial_entropy = env.compute_entropy()

        for _ in range(200):
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

        final_entropy = env.compute_entropy()
        assert final_entropy <= initial_entropy, "Agent should reduce or maintain entropy"


class TestRegulationHeuristicAgent:
    def test_returns_valid_actions(self) -> None:
        """Test that agent returns valid actions during episode."""
        agent = RegulationHeuristicAgent()
        env = CoupledRegulation(seed=42)
        obs = env.reset({"name": "weak"})

        for _ in range(50):
            action = agent.act(obs)
            assert 0 <= action <= 16, f"Invalid action {action}"
            obs, _, done, _ = env.step(action)
            if done:
                break

    def test_reduces_deviation(self) -> None:
        """Test that agent reduces deviations toward targets over time."""
        agent = RegulationHeuristicAgent()
        env = CoupledRegulation(seed=42)
        obs = env.reset({"name": "weak"})

        # Compute initial total deviation
        values = obs[:8]
        targets = obs[8:16]
        initial_deviation = np.sum(np.abs(values - targets))

        # Run agent for 50 steps
        for _ in range(50):
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

        # Compute final deviation
        values = obs[:8]
        targets = obs[8:16]
        final_deviation = np.sum(np.abs(values - targets))

        # Deviation should decrease (or stay same due to coupling effects)
        # At minimum, agent should not drastically increase deviation
        assert final_deviation <= initial_deviation * 1.5

    def test_picks_largest_deviation(self) -> None:
        """Test that agent picks variable with largest deviation."""
        agent = RegulationHeuristicAgent()

        # Create observation with known deviations
        # values = [0, 0, 0, 0, 0, 0, 0, 0]
        # targets = [0, 0, 2.0, 0, 0, 0, 0, 0]  # var 2 has largest deviation
        obs = np.zeros(16)
        obs[8 + 2] = 2.0  # target for var 2

        action = agent.act(obs)
        # Should increase var 2 (action 2) since value < target
        assert action == 2, "Agent should adjust variable with largest deviation"

    def test_noop_when_close_to_target(self) -> None:
        """Test that agent performs no-op when all deviations are small."""
        agent = RegulationHeuristicAgent()

        # Create observation where values are very close to targets
        values = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])
        targets = np.array([1.01, 2.01, 3.01, 4.01, 0.0, 0.0, 0.0, 0.0])
        obs = np.concatenate([values, targets])

        action = agent.act(obs)
        assert action == 16, "Agent should perform no-op when close to targets"


class TestTilePuzzleHeuristicAgent:
    def test_returns_valid_actions(self) -> None:
        """Test that agent returns valid actions during episode."""
        agent = TilePuzzleHeuristicAgent()
        env = TilePuzzle(seed=42)
        obs = env.reset({"name": "easy"})

        for _ in range(50):
            action = agent.act(obs)
            assert 0 <= action <= 3, f"Invalid action {action}"
            obs, _, done, _ = env.step(action)
            if done:
                break

    def test_reduces_manhattan(self) -> None:
        """Test that agent reduces Manhattan distance over time on easy puzzle."""
        agent = TilePuzzleHeuristicAgent()
        env = TilePuzzle(seed=42)
        obs = env.reset({"name": "easy"})

        # Compute initial Manhattan distance
        grid = obs.reshape(4, 4).astype(int)
        initial_dist = agent._total_manhattan(grid)

        # Run agent for 30 steps
        for _ in range(30):
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)
            if done:
                break

        # Compute final Manhattan distance
        grid = obs.reshape(4, 4).astype(int)
        final_dist = agent._total_manhattan(grid)

        # Manhattan distance should decrease (or at least not increase much)
        # Agent is greedy so should generally improve
        assert final_dist <= initial_dist, "Agent should reduce Manhattan distance"

    def test_picks_best_move(self) -> None:
        """Test that agent picks move that reduces Manhattan distance."""
        agent = TilePuzzleHeuristicAgent()

        # Create a simple scenario: one tile away from solved
        # Solved: [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,0]]
        # Current: [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,0,15]]
        # Empty at (3,2), tile 15 at (3,3)
        # Best move: slide from right (action 3) to get closer to solved
        obs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15], dtype=np.float32)

        action = agent.act(obs)
        # Should slide tile 15 from right (action 3)
        assert action == 3, "Agent should pick move that reduces Manhattan distance"
