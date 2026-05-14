import numpy as np
from collections import deque
from typing import Optional, Tuple
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    WINDOW_SIZE, DRIFT_MEASURE,
    PSI_BINS, KL_BINS, KL_EPSILON,
    ADWIN_DELTA, DDM_WARNING_LEVEL, DDM_DRIFT_LEVEL,
    UNCERTAINTY_BINS
)


# ================================================================== #
#  DRIFT DETECTORS                                                     #
# ================================================================== #

class ADWINDetector:
    """
    Simplified ADWIN: theo dõi error rate trong sliding window,
    phát hiện khi mean của nửa cuối khác nửa đầu có ý nghĩa thống kê.
    Output: magnitude = |mean_new - mean_old|
    """

    def __init__(self, delta: float = ADWIN_DELTA):
        self.delta     = delta
        self.window    = deque()
        self.magnitude = 0.0

    def update(self, error_rate: float) -> float:
        self.window.append(error_rate)

        if len(self.window) < 4:
            self.magnitude = 0.0
            return self.magnitude

        data     = list(self.window)
        mid      = len(data) // 2
        old_half = data[:mid]
        new_half = data[mid:]

        mean_old = np.mean(old_half)
        mean_new = np.mean(new_half)

        n         = len(data)
        threshold = np.sqrt((1 / (2 * n)) * np.log(2 / self.delta))

        self.magnitude = abs(mean_new - mean_old)

        if self.magnitude > threshold:
            self.window = deque(new_half)

        return self.magnitude

    def reset(self) -> None:
        self.window.clear()
        self.magnitude = 0.0


class DDMDetector:
    """
    DDM: theo dõi error rate tích lũy và standard deviation.
    Output: 0 (normal), 1 (warning), 2 (drift)
    """

    def __init__(
        self,
        warning_level: float = DDM_WARNING_LEVEL,
        drift_level  : float = DDM_DRIFT_LEVEL
    ):
        self.warning_level = warning_level
        self.drift_level   = drift_level
        self._reset_stats()

    def _reset_stats(self) -> None:
        self.n      = 0
        self.p      = 0.0
        self.s      = 0.0
        self.p_min  = float("inf")
        self.s_min  = float("inf")
        self.status = 0

    def update(self, error_rate: float) -> int:
        self.n += 1
        self.p  = self.p + (error_rate - self.p) / self.n
        self.s  = np.sqrt(self.p * (1 - self.p) / self.n)

        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s

        if self.p + self.s > self.p_min + self.drift_level * self.s_min:
            self._reset_stats()     # reset trước → status về 0
            self.status = 2         # set lại sau → được giữ khi return
        elif self.p + self.s > self.p_min + self.warning_level * self.s_min:
            self.status = 1
        else:
            self.status = 0

        return self.status

    def reset(self) -> None:
        self._reset_stats()


class PSICalculator:
    """
    PSI: so sánh distribution của error rate giữa reference và current window.
    Output: PSI value (liên tục, >= 0)

    Tự giữ internal window của error rates (maxlen = n_bins * 5 batches).
    Window được reset khi set_reference() được gọi (sau mỗi lần retrain)
    → PSI luôn đo drift so với lần retrain gần nhất, không tích lũy lịch sử cũ.
    """

    def __init__(self, n_bins: int = PSI_BINS):
        self.n_bins    = n_bins
        self.reference = None
        self.bin_edges = None
        # Internal window: giữ đủ batches để histogram có ý nghĩa thống kê
        # maxlen = n_bins * 5 là heuristic đảm bảo mỗi bin trung bình 5 điểm
        self._window   = deque(maxlen=n_bins * 5)

    def set_reference(self, error_rates: list) -> None:
        counts, self.bin_edges = np.histogram(error_rates, bins=self.n_bins)
        self.reference = (counts + 1e-6) / (counts + 1e-6).sum()
        # Seed window bằng reference data → PSI bắt đầu gần 0 ngay lập tức
        # và tăng dần khi data thực sự drift, tránh vài batch đầu trả về 0.0
        self._window.clear()
        self._window.extend(error_rates)

    def update(self, error_rate: float) -> float:
        """Nhận scalar error_rate của batch hiện tại, tự quản lý window."""
        self._window.append(error_rate)
        if self.reference is None or len(self._window) < 2:
            return 0.0
        counts, _ = np.histogram(list(self._window), bins=self.bin_edges)
        current   = (counts + 1e-6) / (counts + 1e-6).sum()
        psi = np.sum((current - self.reference) * np.log(current / self.reference))
        return float(max(psi, 0.0))

    def reset(self) -> None:
        self.reference = None
        self.bin_edges = None
        self._window.clear()


class KLDivergenceCalculator:
    """
    Symmetric KL Divergence.
    Output: KL_sym value (liên tục, >= 0)

    Tự giữ internal window, reset khi set_reference() được gọi.
    Cùng design với PSICalculator.
    """

    def __init__(self, n_bins: int = KL_BINS, epsilon: float = KL_EPSILON):
        self.n_bins    = n_bins
        self.epsilon   = epsilon
        self.reference = None
        self.bin_edges = None
        self._window   = deque(maxlen=n_bins * 5)

    def set_reference(self, error_rates: list) -> None:
        counts, self.bin_edges = np.histogram(error_rates, bins=self.n_bins)
        counts         = counts.astype(float) + self.epsilon
        self.reference = counts / counts.sum()
        self._window.clear()
        self._window.extend(error_rates)

    def update(self, error_rate: float) -> float:
        """Nhận scalar error_rate của batch hiện tại, tự quản lý window."""
        self._window.append(error_rate)
        if self.reference is None or len(self._window) < 2:
            return 0.0
        counts  = np.histogram(list(self._window), bins=self.bin_edges)[0].astype(float) + self.epsilon
        current = counts / counts.sum()
        kl_pq   = np.sum(self.reference * np.log(self.reference / current))
        kl_qp   = np.sum(current * np.log(current / self.reference))
        return float((kl_pq + kl_qp) / 2)

    def reset(self) -> None:
        self.reference = None
        self.bin_edges = None
        self._window.clear()


# ================================================================== #
#  STREAM METRICS (main class)                                         #
# ================================================================== #

class StreamMetrics:
    """
    Tính toán và lưu trữ tất cả metrics cần thiết cho:
      - Visualization (notebook)
      - RL Agent state (qua drift_env)
      - Evaluation cuối project

    State vector:
        [rolling_error, error_trend, drift_score, uncertainty, time_since_update]

    Usage:
        metrics = StreamMetrics()
        metrics.set_reference(initial_error_rates)

        for X_batch, y_batch, idx in loader.stream_batches():
            preds      = clf.predict(X_batch)
            y_proba    = clf.predict_proba(X_batch)
            error_rate = clf.get_error_rate(X_batch, y_batch)
            metrics.update(y_batch.values, preds, error_rate, y_proba)

            state = metrics.get_state(time_since_update)
    """

    def __init__(self, window_size: int = WINDOW_SIZE):
        self.window_size = window_size

        # History để visualize
        self.error_rate_history      : list[float] = []
        self.rolling_error_history   : list[float] = []
        self.prequential_acc_history : list[float] = []
        self.drift_measure_history   : list[float] = []
        self.uncertainty_history     : list[float] = []

        # Rolling windows
        self._y_true_window     = deque(maxlen=window_size)
        self._y_pred_window     = deque(maxlen=window_size)
        self._proba_window      = deque(maxlen=window_size)
        self._error_rate_window = deque(maxlen=window_size)

        # Prequential
        self._total_correct = 0
        self._total_seen    = 0

        # Drift detectors
        self._adwin = ADWINDetector()
        self._ddm   = DDMDetector()
        self._psi   = PSICalculator()
        self._kl    = KLDivergenceCalculator()

        # State discretization bins
        self._error_bins       = [0.0, 0.20, 0.35, 0.50, 0.65, 1.0]       # 5 bins
        self._trend_bins       = [-1.0, -0.02, 0.02, 1.0]                  # 3 bins
        self._uncertainty_bins = UNCERTAINTY_BINS                           # từ config
        self._time_bins        = [0, 10, 30, 60, float("inf")]             # 4 bins

    # ------------------------------------------------------------------ #
    #  SETUP                                                               #
    # ------------------------------------------------------------------ #

    def set_reference(self, initial_error_rates: list) -> None:
        """
        Thiết lập reference distribution cho PSI và KL.
        Gọi một lần sau khi train model lần đầu.
        """
        self._psi.set_reference(initial_error_rates)
        self._kl.set_reference(initial_error_rates)
        
    # ------------------------------------------------------------------ #
    #  UPDATE                                                              #
    # ------------------------------------------------------------------ #

    def update(
        self,
        y_true     : np.ndarray,
        y_pred     : np.ndarray,
        error_rate : float,
        y_proba    : np.ndarray,
    ) -> None:
        """
        Cập nhật tất cả metrics sau mỗi batch.

        Args:
            y_true     : true labels của batch
            y_pred     : predicted labels của batch
            error_rate : error rate của batch (từ classifier)
            y_proba    : predicted probabilities của class 1 (từ classifier)
        """
        # --- Rolling window predictions ---
        self._y_true_window.extend(y_true)
        self._y_pred_window.extend(y_pred)
        self._proba_window.extend(y_proba)
        self._error_rate_window.append(error_rate)

        # --- Prequential accuracy ---
        self._total_correct += int(np.sum(y_true == y_pred))
        self._total_seen    += len(y_true)

        # --- Error rate history ---
        self.error_rate_history.append(error_rate)

        # --- Rolling error ---
        rolling_error = 1.0 - float(np.mean(
            np.array(self._y_true_window) == np.array(self._y_pred_window)
        ))
        self.rolling_error_history.append(rolling_error)

        # --- Prequential accuracy ---
        self.prequential_acc_history.append(
            self._total_correct / self._total_seen
        )

        # --- Uncertainty: mean(1 - |2p - 1|) ---
        proba_arr   = np.array(self._proba_window)
        uncertainty = float(np.mean(1.0 - np.abs(2.0 * proba_arr - 1.0)))
        self.uncertainty_history.append(uncertainty)

        # --- Drift measure ---
        drift_value = self._compute_drift(error_rate)
        self.drift_measure_history.append(drift_value)

    def _compute_drift(self, error_rate: float) -> float:
        """Tính drift measure theo DRIFT_MEASURE trong config."""
        if DRIFT_MEASURE == "adwin":
            return self._adwin.update(error_rate)
        elif DRIFT_MEASURE == "ddm":
            return float(self._ddm.update(error_rate))
        elif DRIFT_MEASURE == "psi":
            return self._psi.update(error_rate)     # scalar, PSI tự quản lý window
        elif DRIFT_MEASURE == "kl":
            return self._kl.update(error_rate)      # scalar, KL tự quản lý window
        else:
            raise ValueError(
                f"DRIFT_MEASURE='{DRIFT_MEASURE}' không hợp lệ. "
                f"Chọn: 'adwin' | 'ddm' | 'psi' | 'kl'"
            )

    # ------------------------------------------------------------------ #
    #  GET METRICS                                                         #
    # ------------------------------------------------------------------ #

    def get_rolling_error(self) -> float:
        """Rolling error rate trong window gần nhất."""
        if not self.rolling_error_history:
            return 0.0
        return self.rolling_error_history[-1]

    def get_prequential_accuracy(self) -> float:
        """Prequential accuracy tích lũy từ đầu."""
        if self._total_seen == 0:
            return 0.0
        return self._total_correct / self._total_seen

    def get_error_trend(self, lookback: int = 5) -> float:
        """
        Error trend: rolling_error hiện tại - rolling_error N batches trước.
        Dương = đang tệ đi, âm = đang tốt lên.
        """
        if len(self.rolling_error_history) < lookback + 1:
            return 0.0
        current  = self.rolling_error_history[-1]
        previous = self.rolling_error_history[-lookback - 1]
        return float(current - previous)

    def get_drift_measure(self) -> float:
        """Drift measure hiện tại."""
        if not self.drift_measure_history:
            return 0.0
        return self.drift_measure_history[-1]

    def get_uncertainty(self) -> float:
        """Model uncertainty hiện tại."""
        if not self.uncertainty_history:
            return 0.0
        return self.uncertainty_history[-1]

    # ------------------------------------------------------------------ #
    #  STATE FOR RL AGENT                                                  #
    # ------------------------------------------------------------------ #

    def get_state(self, time_since_update: int) -> Tuple[int, int, int, int, int]:
        """
        Trả về state đã discretize cho RL Agent.

        Args:
            time_since_update: số batches kể từ lần update cuối (từ drift_env)

        Returns:
            Tuple[int, int, int, int, int]:
                (rolling_error_bin, error_trend_bin, drift_bin,
                 uncertainty_bin, time_bin)
        """
        rolling_error = self.get_rolling_error()
        error_trend   = self.get_error_trend()
        drift         = self.get_drift_measure()
        uncertainty   = self.get_uncertainty()

        error_bin       = self._discretize(rolling_error,          self._error_bins)
        trend_bin       = self._discretize(error_trend,            self._trend_bins)
        drift_bin       = self._discretize_drift(drift)
        uncertainty_bin = self._discretize(uncertainty,            self._uncertainty_bins)
        time_bin        = self._discretize(float(time_since_update), self._time_bins)

        return (error_bin, trend_bin, drift_bin, uncertainty_bin, time_bin)

    def _discretize(self, value: float, bins: list) -> int:
        """Chuyển giá trị liên tục thành bin index."""
        for i in range(len(bins) - 1):
            if value < bins[i + 1]:
                return i
        return len(bins) - 2

    def _discretize_drift(self, value: float) -> int:
        """Discretize drift measure – bins khác nhau tùy theo measure."""
        if DRIFT_MEASURE == "ddm":
            return int(value)                                           # Đã là 0, 1, 2
        elif DRIFT_MEASURE == "adwin":
            bins = [0.0, 0.05, 0.15, 0.30, 1.0]                       # 4 bins
        elif DRIFT_MEASURE == "psi":
            bins = [0.0, 0.10, 0.20, 0.35, float("inf")]              # 4 bins
        elif DRIFT_MEASURE == "kl":
            bins = [0.0, 0.10, 0.25, 0.50, float("inf")]              # 4 bins
        return self._discretize(value, bins)

    # ------------------------------------------------------------------ #
    #  SUMMARY                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> dict:
        """Trả về tóm tắt metrics hiện tại."""
        return {
            "prequential_accuracy" : round(self.get_prequential_accuracy(), 4),
            "rolling_error"        : round(self.get_rolling_error(), 4),
            "error_trend"          : round(self.get_error_trend(), 4),
            "drift_measure"        : round(self.get_drift_measure(), 4),
            "uncertainty"          : round(self.get_uncertainty(), 4),
            "n_batches_seen"       : len(self.error_rate_history),
        }

    def reset(self) -> None:
        """Reset toàn bộ metrics – dùng khi bắt đầu experiment mới."""
        self.__init__(self.window_size)


# ================================================================== #
#  QUICK TEST                                                          #
# ================================================================== #

if __name__ == "__main__":
    from utils.data_loader import AirlinesDataLoader
    from models.LightGBM import LightGBM

    loader = AirlinesDataLoader()
    X_train, y_train = loader.get_initial_train_data()

    clf = LightGBM()
    clf.train(X_train, y_train)

    metrics = StreamMetrics()

    # Set reference từ initial train set
    init_errors = [
        clf.get_error_rate(
            X_train.iloc[i*500:(i+1)*500],
            y_train.iloc[i*500:(i+1)*500]
        ) for i in range(10)
    ]
    metrics.set_reference(init_errors)

    # Stream
    print(f"\n{'Batch':>6} | {'RollErr':>7} | {'Trend':>7} | "
          f"{'Drift':>7} | {'Uncert':>7} | State")
    print("-" * 70)

    for X_batch, y_batch, idx in loader.stream_batches():
        preds      = clf.predict(X_batch)
        y_proba    = clf.predict_proba(X_batch)
        error_rate = clf.get_error_rate(X_batch, y_batch)
        metrics.update(y_batch.values, preds, error_rate, y_proba)

        if idx % 20 == 0:
            state = metrics.get_state(time_since_update=idx)
            s     = metrics.summary()
            print(f"{idx:>6} | {s['rolling_error']:>7.3f} | "
                  f"{s['error_trend']:>7.4f} | "
                  f"{s['drift_measure']:>7.4f} | "
                  f"{s['uncertainty']:>7.4f} | {state}")