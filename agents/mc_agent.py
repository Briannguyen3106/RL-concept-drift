import numpy as np
from typing import Optional
import sys
import os
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MC_GAMMA, MC_ALPHA, GRADIENT_BANDIT_BETA, MC_TEMP_START, MC_TEMP_END, MC_TEMP_DECAY, MC_EPISODE_STARTS
from agents.base_agent import BaseAgent
from environment.drift_env import DriftStreamEnv, GradientBandit, ACTIONS


class MCAgent(BaseAgent):
    """
    Monte Carlo Agent với Every-visit MC và Gradient Bandit exploration.

    Thiết kế:
        - Every-visit: update Q(s,a) mỗi lần (s,a) xuất hiện trong episode
        - GradientBandit: select action theo softmax policy π(a|s)
        - Update H(s,a) dựa trên return G_t thay vì immediate reward R_t
          → Agent học long-term consequences của actions
        - Baseline G_bar update với β cố định (exponential recency-weighted)
          → Quên dần returns cũ, phù hợp với non-stationary environment

    Update rules:
        G_t    = r_t + γ·G_{t+1}                          (backward)
        δ      = G_t - G_bar(s_t)                          (advantage)
        H(s,a) += α·δ·(1 - π(a|s))   for a = A_t          (chosen action)
        H(s,b) -= α·δ·π(b|s)         for b ≠ A_t          (other actions)
        G_bar  += β·(G_t - G_bar)                          (baseline update)
    """

    def __init__(self):
        super().__init__(
            n_actions = 5,
            name      = "MCAgent"
        )
        self.gamma = MC_GAMMA
        self.alpha = MC_ALPHA
        self.beta  = GRADIENT_BANDIT_BETA
        self.temp  = MC_TEMP_START

        # Gradient Bandit preferences H[state] = np.ndarray(n_actions)
        self.H     : dict[tuple, np.ndarray] = {}

        # Baseline G_bar per state – exponential recency-weighted
        self.G_bar : dict[tuple, float]      = {}

    # ------------------------------------------------------------------ #
    #  PRIVATE                                                             #
    # ------------------------------------------------------------------ #

    def _get_H(self, state: tuple) -> np.ndarray:
        """Lazy init preference vector cho state."""
        if state not in self.H:
            self.H[state]     = np.zeros(self.n_actions)
            self.G_bar[state] = 0.0
        return self.H[state]

    def _softmax(self, h: np.ndarray, temp: float = None) -> np.ndarray:
        """
        Numerically stable softmax với temperature.
        temp lớn → uniform (explore nhiều)
        temp nhỏ → greedy (exploit nhiều)
        """
        t     = temp if temp is not None else self.temp
        exp_h = np.exp((h - np.max(h)) / t)
        return exp_h / exp_h.sum()

    def _compute_returns(
        self,
        rewards: list[float]
    ) -> list[float]:
        """
        Tính discounted returns theo backward pass.

        G_T   = 0
        G_t   = r_t + γ·G_{t+1}

        Args:
            rewards: list rewards [r_0, r_1, ..., r_T]

        Returns:
            list returns [G_0, G_1, ..., G_T]
        """
        T       = len(rewards)
        returns = [0.0] * T
        G       = 0.0

        for t in reversed(range(T)):
            G          = rewards[t] + self.gamma * G
            returns[t] = G

        return returns

    # ------------------------------------------------------------------ #
    #  SELECT ACTION                                                       #
    # ------------------------------------------------------------------ #

    def select_action(self, state: tuple):
        """
        Sample action theo softmax policy π(a|s).

        Returns:
            action : int
            pi     : np.ndarray – policy distribution (cần cho update)
        """
        h  = self._get_H(state)
        pi = self._softmax(h)
        action = int(np.random.choice(self.n_actions, p=pi))
        return action, pi

    # ------------------------------------------------------------------ #
    #  UPDATE                                                              #
    # ------------------------------------------------------------------ #

    def update(self, episode_buffer: list[tuple]) -> None:
        """
        Every-visit MC update sau khi kết thúc episode.

        Args:
            episode_buffer: list of (state, action, reward, pi)
                            thu thập trong suốt episode

        Update rules:
            G_t    = r_t + γ·G_{t+1}
            δ      = G_t - G_bar(s_t)
            H(A_t) += α·δ·(1 - π(A_t|s_t))
            H(b)   -= α·δ·π(b|s_t)   ∀b ≠ A_t
            G_bar  += β·(G_t - G_bar)
        """
        states, actions, rewards, pis = zip(*episode_buffer)

        # Tính returns cho toàn bộ episode
        returns = self._compute_returns(list(rewards))

        # Every-visit: update mỗi lần (s,a) xuất hiện
        for t, (state, action, G_t, pi) in enumerate(
            zip(states, actions, returns, pis)
        ):
            self._get_H(state)      # Đảm bảo state đã init

            # Advantage: G_t so với baseline
            g_bar = self.G_bar[state]
            delta = G_t - g_bar

            # Cập nhật preferences theo policy gradient
            self.H[state]        -= self.alpha * delta * pi         # tất cả actions
            self.H[state][action] += self.alpha * delta             # sửa action được chọn

            # Cập nhật baseline với β cố định
            self.G_bar[state] = g_bar + self.beta * delta

            # Cập nhật Q-values để BaseAgent.get_policy() hoạt động
            self.Q[state] = self._softmax(self.H[state])

    # ------------------------------------------------------------------ #
    #  RUN EPISODE                                                         #
    # ------------------------------------------------------------------ #

    def run_episode(self, env: DriftStreamEnv, ep: int = 0) -> dict:
        """
        Chạy 1 episode hoàn chỉnh theo Every-visit MC.

        Flow:
            1. Reset environment (từ start_batch)
            2. Thu thập trajectory (state, action, reward, pi)
            3. Tính returns G_t theo backward pass
            4. Update H(s,a) theo policy gradient với G_t

        Args:
            env        : DriftStreamEnv instance
            start_batch: batch index để bắt đầu episode (Hướng B)

        Returns:
            dict chứa metrics của episode
        """
        start_batch = MC_EPISODE_STARTS[ep % len(MC_EPISODE_STARTS)]
        # 1. Reset environment – fresh model mỗi episode
        state          = env.reset(start_batch=start_batch)
        episode_buffer = []
        total_reward   = 0.0
        action_counts  = {k: 0 for k in ACTIONS.values()}
        info = {}

        # 2. Thu thập trajectory
        while not env.is_done:
            # Select action theo softmax policy
            action, pi = self.select_action(state)

            # Step – tắt auto-update explorer vì MC tự update cuối episode
            next_state, reward, done, info = env.step(
                action,
                current_state   = state,
                pi             = pi,
                update_explorer = False     # MC tự update sau episode
            )
            
            if done and "action_taken" not in info:
                state = next_state
                break

            episode_buffer.append((state, action, reward, pi))
            total_reward  += reward
            action_counts[info["action_taken"]] += 1
            state = next_state

        # 3 & 4. Tính returns và update H sau khi episode kết thúc
        if episode_buffer:
            self.update(episode_buffer)

        # Decay temperature sau mỗi episode
        self.temp = max(MC_TEMP_END, self.temp * MC_TEMP_DECAY)

        return {
            "total_reward" : total_reward,
            "n_steps"      : len(episode_buffer),
            "action_counts": action_counts,
            "final_info"   : info,
        }

    # ------------------------------------------------------------------ #
    #  RESET                                                               #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset toàn bộ learning – kể cả H và G_bar."""
        super().reset()
        self.H.clear()
        self.G_bar.clear()

    #======================================================================
    #Save and load
    #======================================================================
    def save(self, path: str) -> None:
        """Lưu agent state ra file."""
        with open(path, 'wb') as f:
            pickle.dump({
                'H'               : self.H,
                'G_bar'           : self.G_bar,
                'Q'               : self.Q,
                'temp'            : self.temp,
                'episode_rewards' : self.episode_rewards,
                'episode_lengths' : self.episode_lengths,
                'action_counts'   : self.action_counts,
            }, f)
        print(f'[MCAgent] Saved → {path}')
    
    @classmethod
    def load(cls, path: str) -> 'MCAgent':
        """Load agent từ file, trả về MCAgent instance."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        agent = cls()
        agent.H               = data['H']
        agent.G_bar           = data['G_bar']
        agent.Q               = data['Q']
        agent.temp            = data['temp']
        agent.episode_rewards = data['episode_rewards']
        agent.episode_lengths = data['episode_lengths']
        agent.action_counts   = data['action_counts']
        print(f'[MCAgent] Loaded | Episodes: {len(agent.episode_rewards)} | States: {len(agent.Q)}')
        return agent

    


# ================================================================== #
#  QUICK TEST                                                          #
# ================================================================== #

if __name__ == "__main__":
    from utils.data_loader import AirlinesDataLoader
    from models.LightGBM import LightGBM
    from metrics.stream_metrics import StreamMetrics
    from environment.drift_env import DriftStreamEnv

    loader  = AirlinesDataLoader()
    clf     = LightGBM()
    metrics = StreamMetrics()
    env     = DriftStreamEnv(loader, clf, metrics)
    agent   = MCAgent()

    print(f"MC Agent: {agent}")
    print(f"Gamma: {agent.gamma}, Alpha: {agent.alpha}, Beta: {agent.beta}")

    # Train 3 episodes để test
    history = agent.train(env, n_episodes=3)

    print(f"\nAfter training:")
    print(f"  States visited : {len(agent.Q)}")
    print(f"  Episodes       : {len(agent.episode_rewards)}")
    print(f"  Avg reward     : {np.mean(history['episode_rewards']):.4f}")
    print(f"  Action counts  : {history['action_counts']}")