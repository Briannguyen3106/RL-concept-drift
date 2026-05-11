import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RANDOM_SEED, MC_EPISODE_STARTS


class BaseAgent(ABC):
    """
    Abstract base class cho tất cả RL agents.

    Mỗi agent con phải implement:
        - select_action(state) → int
        - update(...)          → None
        - run_episode(env)     → dict

    BaseAgent cung cấp:
        - train(env, n_episodes) → dict  (chung cho tất cả agents)
        - get_policy()           → dict
        - get_q_values(state)    → np.ndarray
        - reset()                → None

    Design:
        BaseAgent không giữ bất kỳ logic exploration nào –
        mỗi agent con tự quyết định cách explore (GradientBandit,
        epsilon-greedy, v.v.) vì MC và TD explore khác nhau.
    """

    def __init__(self, n_actions: int, name: str = "BaseAgent"):
        self.n_actions = n_actions
        self.name      = name
        np.random.seed(RANDOM_SEED)

        # Q-table: dict[state → np.ndarray shape (n_actions,)]
        # Dùng dict thay vì array vì state space có thể sparse
        self.Q: dict[tuple, np.ndarray] = {}

        # Tracking per episode
        self.episode_rewards  : list[float] = []   # Total reward mỗi episode
        self.episode_lengths  : list[int]   = []   # Số steps mỗi episode
        self.action_counts    : dict[str, int] = { # Tổng hợp actions
            "no_action"     : 0,
            "partial_update": 0,
            "full_retrain"  : 0,
            "alert"         : 0,
            "switch_model"  : 0,
        }

    # ------------------------------------------------------------------ #
    #  ABSTRACT – mỗi agent con phải implement                            #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def select_action(self, state: tuple) -> int:
        """
        Chọn action tại state hiện tại.

        Args:
            state: tuple đã discretize từ StreamMetrics.get_state()

        Returns:
            action index (0-4)
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Cập nhật Q-values / preferences.
        Interface khác nhau tùy agent:
            MC : update(episode_buffer)
            TD : update(state, action, reward, next_state)
        """
        pass

    @abstractmethod
    def run_episode(self, env) -> dict:
        """
        Chạy 1 episode hoàn chỉnh.

        Returns:
            dict chứa metrics của episode:
            {
                "total_reward"  : float,
                "n_steps"       : int,
                "action_counts" : dict,
                "final_info"    : dict,
            }
        """
        pass

    # ------------------------------------------------------------------ #
    #  CONCRETE – dùng chung cho tất cả agents                            #
    # ------------------------------------------------------------------ #

    def train(self, env, n_episodes: int, verbose: bool = True) -> dict:
        """
        Chạy nhiều episodes, thu thập training history.

        Args:
            env        : DriftStreamEnv instance
            n_episodes : số episodes để train
            verbose    : in progress mỗi episode

        Returns:
            training_history dict:
            {
                "episode_rewards" : list[float],
                "episode_lengths" : list[int],
                "action_counts"   : dict,
            }
        """
        print(f"\n{'='*60}")
        print(f"Training {self.name} for {n_episodes} episodes")
        print(f"{'='*60}")

        for ep in range(n_episodes):
            start_batch = MC_EPISODE_STARTS[ep % len(MC_EPISODE_STARTS)]
            result = self.run_episode(env, start_batch=start_batch)

            self.episode_rewards.append(result["total_reward"])
            self.episode_lengths.append(result["n_steps"])

            # Cộng dồn action counts
            for action, count in result["action_counts"].items():
                self.action_counts[action] += count

            if verbose:
                avg_reward = np.mean(self.episode_rewards[-10:])  # Moving avg 10 eps
                print(
                    f"Episode {ep+1:>4}/{n_episodes} | "
                    f"Reward: {result['total_reward']:>8.2f} | "
                    f"Avg(10): {avg_reward:>8.2f} | "
                    f"Steps: {result['n_steps']:>4}"
                )

        print(f"\nTraining complete!")
        print(f"  Final avg reward (last 10): {np.mean(self.episode_rewards[-10:]):.4f}")
        print(f"  Action distribution: {self.action_counts}")

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "action_counts"  : self.action_counts,
        }

    def get_q_values(self, state: tuple) -> np.ndarray:
        """
        Trả về Q-values tại state.
        Khởi tạo 0 nếu state chưa được thăm.
        """
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]

    def get_policy(self) -> dict:
        """
        Trả về greedy policy hiện tại.

        Returns:
            dict[state → best_action_index]
        """
        return {
            state: int(np.argmax(q_vals))
            for state, q_vals in self.Q.items()
        }

    def get_best_action(self, state: tuple) -> int:
        """Trả về greedy action tại state (argmax Q)."""
        return int(np.argmax(self.get_q_values(state)))

    def reset(self) -> None:
        """Reset toàn bộ learning history – dùng khi train lại từ đầu."""
        self.Q               = {}
        self.episode_rewards  = []
        self.episode_lengths  = []
        self.action_counts    = {k: 0 for k in self.action_counts}

    # ------------------------------------------------------------------ #
    #  PROPERTIES                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_episodes_trained(self) -> int:
        return len(self.episode_rewards)

    @property
    def n_states_visited(self) -> int:
        """Số states đã được thăm ít nhất 1 lần."""
        return len(self.Q)

    def __repr__(self) -> str:
        return (
            f"{self.name}("
            f"episodes={self.n_episodes_trained}, "
            f"states_visited={self.n_states_visited})"
        )