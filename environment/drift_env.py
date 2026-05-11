import numpy as np
import pandas as pd
import copy
from collections import deque
from typing import Tuple, Optional
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    CATEGORICAL_COLS,
    INITIAL_TRAIN_SIZE, BATCH_SIZE,
    FULL_WINDOW_BATCHES, PARTIAL_WINDOW_BATCHES,
    CHECKPOINT_THRESHOLD,
    REWARD_ACCURACY_WEIGHT, ACTION_COSTS,
    DRIFT_MISS_PENALTY, DRIFT_MISS_THRESHOLD,
    EXPLORATION_STRATEGY,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    GRADIENT_BANDIT_ALPHA, GRADIENT_BANDIT_BETA,
)
from utils.data_loader import AirlinesDataLoader
from models.LightGBM import LightGBM
from metrics.stream_metrics import StreamMetrics


# ================================================================== #
#  ACTION SPACE                                                        #
# ================================================================== #

ACTIONS = {
    0: "no_action",
    1: "partial_update",
    2: "full_retrain",
    3: "alert",
    4: "switch_model",
}
N_ACTIONS = len(ACTIONS)


# ================================================================== #
#  EXPLORATION STRATEGIES                                              #
# ================================================================== #

class EpsilonGreedy:
    """
    Epsilon-greedy exploration với epsilon decay theo episodes.
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.epsilon   = EPSILON_START

    def select_action(self, q_values: np.ndarray) -> int:
        """
        Chọn action theo epsilon-greedy.

        Args:
            q_values: Q(s, :) – array shape (n_actions,)

        Returns:
            action index
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)        # Explore
        return int(np.argmax(q_values))                     # Exploit

    def decay(self) -> None:
        """Gọi sau mỗi episode để giảm epsilon."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def reset(self) -> None:
        self.epsilon = EPSILON_START


class GradientBandit:
    """
    Gradient Bandit với softmax policy và incremental baseline.

    Preference update:
        H(A_t) ← H(A_t) + α(R_t - R_bar)(1 - π(A_t))
        H(b)   ← H(b)   - α(R_t - R_bar)π(b)   ∀b ≠ A_t

    Baseline update (β cố định để quên reward cũ):
        R_bar ← R_bar + β(R_t - R_bar)
    """

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.alpha     = GRADIENT_BANDIT_ALPHA
        self.beta      = GRADIENT_BANDIT_BETA
        self.H         = {}     # H[state] = np.ndarray shape (n_actions,)
        self.R_bar     = {}     # R_bar[state] = float (baseline per state)

    def _get_H(self, state: tuple) -> np.ndarray:
        """Lấy preference vector cho state, khởi tạo 0 nếu chưa có."""
        if state not in self.H:
            self.H[state]     = np.zeros(self.n_actions)
            self.R_bar[state] = 0.0
        return self.H[state]

    def _softmax(self, h: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        exp_h = np.exp(h - np.max(h))
        return exp_h / exp_h.sum()

    def select_action(self, state: tuple) -> Tuple[int, np.ndarray]:
        """
        Sample action theo softmax policy.

        Returns:
            action   : int
            pi       : np.ndarray – policy distribution (dùng cho update)
        """
        h  = self._get_H(state)
        pi = self._softmax(h)
        action = int(np.random.choice(self.n_actions, p=pi))
        return action, pi

    def update(
        self,
        state  : tuple,
        action : int,
        reward : float,
        pi     : np.ndarray
    ) -> None:
        """
        Cập nhật preferences theo policy gradient.

        H(A_t) ← H(A_t) + α(R - R_bar)(1 - π(A_t))
        H(b)   ← H(b)   - α(R - R_bar)π(b)   ∀b ≠ A_t

        Baseline:
        R_bar  ← R_bar + β(R - R_bar)
        """
        self._get_H(state)      # Đảm bảo state đã được khởi tạo
        r_bar  = self.R_bar[state]
        delta  = reward - r_bar

        # Cập nhật tất cả preferences
        self.H[state] -= self.alpha * delta * pi                # H(b) cho b ≠ A_t
        self.H[state][action] += self.alpha * delta             # Sửa lại H(A_t)

        # Cập nhật baseline với β cố định
        self.R_bar[state] = r_bar + self.beta * delta

    def reset(self) -> None:
        """Reset preferences – gọi khi bắt đầu training mới."""
        self.H.clear()
        self.R_bar.clear()


# ================================================================== #
#  DRIFT STREAM ENVIRONMENT                                            #
# ================================================================== #

class DriftStreamEnv:
    """
    RL Environment cho bài toán Adaptive Concept Drift Handling.

    Interface (Gym-like):
        state          = env.reset()
        state, reward, done, info = env.step(action)

    Quản lý:
        - Data stream (qua AirlinesDataLoader)
        - Base classifier (LightGBM)
        - Stream metrics (rolling error, drift, uncertainty...)
        - Window buffer cho retrain
        - Checkpoint model tốt nhất
        - Exploration strategy (epsilon-greedy hoặc gradient bandit)

    Usage:
        loader  = AirlinesDataLoader()
        clf     = BaseClassifier()
        metrics = StreamMetrics()
        env     = DriftStreamEnv(loader, clf, metrics)

        state = env.reset()
        while True:
            action, pi = env.select_action(state)
            next_state, reward, done, info = env.step(action, pi)
            if done:
                break
            state = next_state
    """

    def __init__(
        self,
        loader  : AirlinesDataLoader,
        clf     : LightGBM,
        metrics : StreamMetrics,
    ):
        self.loader  = loader
        self.clf     = clf
        self.metrics = metrics

        # Window buffer lưu (X_batch, y_batch) gần nhất
        self._window_buffer: deque = deque(maxlen=FULL_WINDOW_BATCHES)

        # Checkpoint
        self._best_model   : Optional[LightGBM] = None
        self._best_accuracy: float = 0.0

        # Internal state
        self._time_since_update: int  = 0
        self._alert_active     : bool = False
        self._batch_iterator          = None
        self._current_batch_idx: int  = 0
        self._done             : bool = False

        # Exploration strategy
        self.explorer = self._build_explorer()

    # ------------------------------------------------------------------ #
    #  SETUP                                                               #
    # ------------------------------------------------------------------ #

    def _build_explorer(self):
        if EXPLORATION_STRATEGY == "epsilon_greedy":
            return EpsilonGreedy(N_ACTIONS)
        elif EXPLORATION_STRATEGY == "gradient_bandit":
            return GradientBandit(N_ACTIONS)
        else:
            raise ValueError(
                f"EXPLORATION_STRATEGY='{EXPLORATION_STRATEGY}' không hợp lệ. "
                f"Chọn: 'epsilon_greedy' | 'gradient_bandit'"
            )

    # ------------------------------------------------------------------ #
    #  RESET                                                               #
    # ------------------------------------------------------------------ #

    def reset(self, start_batch: int = 0, mode: str = "train") -> tuple:
        """
        Reset environment cho episode mới.
        - Retrain classifier từ đầu trên initial train data
        - Reset metrics, buffer, internal state
        - Set reference distribution cho PSI/KL
        - Skip đến start_batch nếu > 0 (chỉ áp dụng khi mode="train")

        Args:
            start_batch: batch index để bắt đầu episode (chỉ dùng khi mode="train")
            mode       : "train" → stream_train_batches (70% đầu)
                         "test"  → stream_test_batches  (30% cuối)
                         Khi mode="test", start_batch bị bỏ qua vì
                         test stream luôn bắt đầu từ đầu phần test.

        Returns:
            initial state tuple (đã discretize)
        """
        # Fresh model
        X_train, y_train = self.loader.get_initial_train_data()
        self.clf.train(X_train, y_train)

        # Reset metrics
        self.metrics.reset()

        # Set reference distribution cho PSI/KL
        # Tính error rates trên initial train set (10 folds)
        fold_size   = INITIAL_TRAIN_SIZE // 10
        init_errors = [
            self.clf.get_error_rate(
                X_train.iloc[i * fold_size: (i + 1) * fold_size],
                y_train.iloc[i * fold_size: (i + 1) * fold_size]
            )
            for i in range(10)
        ]
        self.metrics.set_reference(init_errors)

        # Reset internal state
        self._window_buffer.clear()
        self._time_since_update = 0
        self._alert_active      = False
        self._best_model        = None
        self._best_accuracy     = 0.0
        self._done              = False
        self._current_batch_idx = 0

        # Chọn batch iterator theo mode
        if mode == "train":
            self._batch_iterator = self.loader.stream_train_batches()
            # Skip đến start_batch nếu cần (rotate starting points khi train)
            if start_batch > 0:
                for _ in range(start_batch):
                    try:
                        next(self._batch_iterator)
                        self._current_batch_idx += 1
                    except StopIteration:
                        break
        elif mode == "test":
            # Test stream luôn chạy toàn bộ từ đầu đến cuối, không skip
            self._batch_iterator = self.loader.stream_test_batches()
        else:
            raise ValueError(f"mode='{mode}' không hợp lệ. Chọn: 'train' | 'test'")

        # Decay epsilon sau mỗi episode (chỉ epsilon-greedy, chỉ khi train)
        if mode == "train" and isinstance(self.explorer, EpsilonGreedy):
            self.explorer.decay()

        return self.metrics.get_state(self._time_since_update)

    # ------------------------------------------------------------------ #
    #  SELECT ACTION                                                       #
    # ------------------------------------------------------------------ #

    def select_action(self, state: tuple):
        """
        Chọn action theo exploration strategy hiện tại.

        Returns:
            action : int
            pi     : np.ndarray | None (chỉ có với GradientBandit)
        """
        if isinstance(self.explorer, EpsilonGreedy):
            # Cần Q-values từ agent – trả về None, agent tự gọi
            # drift_env không giữ Q-table, agent giữ
            raise RuntimeError(
                "Với EpsilonGreedy, agent tự gọi explorer.select_action(q_values). "
                "drift_env.select_action() chỉ dùng với GradientBandit."
            )
        elif isinstance(self.explorer, GradientBandit):
            action, pi = self.explorer.select_action(state)
            return action, pi

    # ------------------------------------------------------------------ #
    #  STEP                                                                #
    # ------------------------------------------------------------------ #

    def step(
        self,
        action          : int,
        pi              : Optional[np.ndarray] = None,
        update_explorer : bool = True,
    ) -> Tuple[tuple, float, bool, dict]:
        """
        Thực thi action, consume batch tiếp theo, tính reward.

        Args:
            action          : int (0-4)
            pi              : softmax policy distribution (chỉ cần với GradientBandit)
            update_explorer : nếu False → không auto-update GradientBandit
                              Dùng False với MC Agent vì MC tự update
                              sau khi tính returns cuối episode

        Returns:
            next_state : tuple (đã discretize)
            reward     : float
            done       : bool
            info       : dict
        """
        if self._done:
            raise RuntimeError("Episode đã kết thúc. Gọi reset() trước.")

        # 1. Lấy batch tiếp theo
        try:
            X_batch, y_batch, batch_idx = next(self._batch_iterator)
            self._current_batch_idx = batch_idx
        except StopIteration:
            self._done = True
            return (
                self.metrics.get_state(self._time_since_update),
                0.0, True,
                {"batch_idx": self._current_batch_idx,
                "action_taken" : "no_action",
                "done": True}
            )

        # 2. Thêm batch vào window buffer
        self._window_buffer.append((X_batch, y_batch))

        # 3. Thực thi action
        action_name      = ACTIONS[action]
        checkpoint_saved = False

        if action_name == "no_action":
            pass

        elif action_name == "partial_update":
            X_w, y_w = self._get_window(PARTIAL_WINDOW_BATCHES)
            self.clf.partial_update(X_w, y_w)
            self._time_since_update = 0
            self._alert_active      = False
            # Reset PSI/KL reference về distribution của data vừa train
            # → PSI đo drift so với model hiện tại, không phải model ban đầu
            self._reset_drift_reference(X_w, y_w)

        elif action_name == "full_retrain":
            X_w, y_w = self._get_window(FULL_WINDOW_BATCHES)
            self.clf.full_retrain(X_w, y_w)
            self._time_since_update = 0
            self._alert_active      = False
            # Reset PSI/KL reference về distribution của data vừa train
            self._reset_drift_reference(X_w, y_w)

        elif action_name == "alert":
            self._alert_active = True   # Đánh dấu, chưa làm gì với model

        elif action_name == "switch_model":
            if self._best_model is not None:
                self.clf.model      = copy.deepcopy(self._best_model.model)
                self.clf.is_trained = True
                self._time_since_update = 0
                self._alert_active      = False
            # Nếu chưa có checkpoint → treat như no_action

        # 4. Predict và update metrics
        preds      = self.clf.predict(X_batch)
        y_proba    = self.clf.predict_proba(X_batch)
        error_rate = self.clf.get_error_rate(X_batch, y_batch)

        self.metrics.update(y_batch.values, preds, error_rate, y_proba)
        self._time_since_update += 1

        # 5. Cập nhật checkpoint
        rolling_acc = 1.0 - self.metrics.get_rolling_error()
        if rolling_acc > self._best_accuracy and rolling_acc >= CHECKPOINT_THRESHOLD:
            self._best_model    = copy.deepcopy(self.clf)
            self._best_accuracy = rolling_acc
            checkpoint_saved    = True

        # 6. Tính reward
        reward = self._compute_reward(error_rate, action_name)

        # 7. Cập nhật GradientBandit nếu đang dùng
        # update_explorer=False khi MC Agent tự quản lý update sau episode
        if update_explorer and isinstance(self.explorer, GradientBandit) and pi is not None:
            next_state = self.metrics.get_state(self._time_since_update)
            self.explorer.update(next_state, action, reward, pi)

        # 8. Build next state và info
        next_state = self.metrics.get_state(self._time_since_update)

        info = {
            "batch_idx"        : batch_idx,
            "error_rate"       : round(error_rate, 4),
            "rolling_error"    : round(self.metrics.get_rolling_error(), 4),
            "drift_measure"    : round(self.metrics.get_drift_measure(), 4),
            "uncertainty"      : round(self.metrics.get_uncertainty(), 4),
            "action_taken"     : action_name,
            "reward"           : round(reward, 4),
            "checkpoint_saved" : checkpoint_saved,
        }

        return next_state, reward, self._done, info

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS                                                     #
    # ------------------------------------------------------------------ #

    def _get_window(self, n_batches: int) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Lấy n_batches gần nhất từ window buffer, concat thành DataFrame.
        Restore category dtype sau khi concat.
        """
        recent = list(self._window_buffer)[-n_batches:]
        X = pd.concat([b[0] for b in recent], ignore_index=True)
        y = pd.concat([b[1] for b in recent], ignore_index=True)

        # Restore category dtype (bị mất sau concat)
        for col in CATEGORICAL_COLS:
            X[col] = X[col].astype("category")

        return X, y

    def _reset_drift_reference(
        self,
        X_w: pd.DataFrame,
        y_w: pd.Series,
        n_folds: int = 5,
    ) -> None:
        """
        Reset PSI/KL reference distribution sau khi retrain.

        Tính error rates của model mới trên window data vừa train (chia n_folds),
        dùng làm reference mới → PSI/KL đo drift so với model hiện tại,
        không phải model ban đầu.

        Args:
            X_w    : features của window data vừa dùng để retrain
            y_w    : labels tương ứng
            n_folds: số folds để tính error rate distribution
        """
        fold_size   = max(len(X_w) // n_folds, 1)
        new_errors  = []
        for i in range(n_folds):
            start = i * fold_size
            end   = start + fold_size
            if start >= len(X_w):
                break
            new_errors.append(
                self.clf.get_error_rate(
                    X_w.iloc[start:end],
                    y_w.iloc[start:end],
                )
            )
        if new_errors:
            self.metrics.set_reference(new_errors)

    def _compute_reward(self, error_rate: float, action_name: str) -> float:
        """
        Tính reward:
            R = R_accuracy - R_cost - R_penalty

        R_accuracy = (1 - error_rate) × REWARD_ACCURACY_WEIGHT
        R_cost     = ACTION_COSTS[action]
        R_penalty  = DRIFT_MISS_PENALTY nếu no_action và error_rate > threshold
        """
        r_accuracy = (1.0 - error_rate) * REWARD_ACCURACY_WEIGHT
        r_cost     = ACTION_COSTS.get(action_name, 0.0)
        r_penalty  = 0.0

        if action_name == "no_action" and error_rate > DRIFT_MISS_THRESHOLD:
            if self._alert_active:
                r_penalty = DRIFT_MISS_PENALTY * 0.5  # Penalty nhẹ hơn
            else:
                r_penalty = DRIFT_MISS_PENALTY
        return r_accuracy - r_cost - r_penalty

    # ------------------------------------------------------------------ #
    #  PROPERTIES                                                          #
    # ------------------------------------------------------------------ #

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    @property
    def action_names(self) -> dict:
        return ACTIONS

    @property
    def is_done(self) -> bool:
        return self._done


# ================================================================== #
#  QUICK TEST                                                          #
# ================================================================== #

if __name__ == "__main__":
    loader  = AirlinesDataLoader()
    clf     = LightGBM()
    metrics = StreamMetrics()
    env     = DriftStreamEnv(loader, clf, metrics)

    print("=== Testing GradientBandit (1 episode) ===\n")
    state = env.reset()
    print(f"Initial state: {state}\n")

    total_reward = 0.0
    step_count   = 0

    print(f"{'Step':>5} | {'Action':>15} | {'Reward':>7} | "
          f"{'RollErr':>7} | {'Drift':>7} | {'Ckpt':>5}")
    print("-" * 65)

    while not env.is_done:
        q_values = np.zeros(env.n_actions)          # random agent để test
        action   = env.explorer.select_action(q_values)
        pi       = None
        next_state, reward, done, info = env.step(action, pi)

        total_reward += reward
        step_count   += 1

        if step_count % 50 == 0 or done:
            if 'reward' not in info:   # StopIteration case
                break
            print(f"{step_count:>5} | {info['action_taken']:>15} | "
                  f"{info['reward']:>7.4f} | "
                  f"{info['rolling_error']:>7.3f} | "
                  f"{info['drift_measure']:>7.4f} | "
                  f"{'✓' if info['checkpoint_saved'] else '':>5}")

        state = next_state

    print(f"\nEpisode done | Steps: {step_count} | "
          f"Total reward: {total_reward:.2f} | "
          f"Avg reward: {total_reward/step_count:.4f}")