import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional
import os
import sys

try:
    import mplcursors
    _HAS_MPLCURSORS = True
except ImportError:
    _HAS_MPLCURSORS = False
    print("[Visualizer] mplcursors không có – hover tooltip bị tắt. "
          "Cài bằng: pip install mplcursors")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR, BATCH_SIZE, DRIFT_MEASURE


# ================================================================== #
#  STYLE DEFAULTS                                                      #
# ================================================================== #

PALETTE = [
    "#2196F3",  # Blue       – Static
    "#4CAF50",  # Green      – Periodic
    "#FF9800",  # Orange     – Always Retrain
    "#9C27B0",  # Purple     – RL Agent (DP)
    "#F44336",  # Red        – RL Agent (MC)
    "#00BCD4",  # Cyan       – RL Agent (SARSA)
    "#FF5722",  # Deep Orange– RL Agent (Q-Learning)
]

plt.rcParams.update({
    "figure.dpi"      : 120,
    "axes.spines.top" : False,
    "axes.spines.right": False,
    "axes.grid"       : True,
    "grid.alpha"      : 0.3,
    "font.size"       : 11,
})


# ================================================================== #
#  HELPER                                                              #
# ================================================================== #

def _save_or_show(fig: plt.Figure, filename: Optional[str]) -> None:
    """Lưu file nếu có filename, luôn show."""
    if filename:
        path = os.path.join(RESULTS_DIR, filename)
        fig.savefig(path, bbox_inches="tight")
        print(f"[Visualizer] Saved → {path}")
    plt.tight_layout()
    plt.show()


def _batch_to_x(history: list) -> np.ndarray:
    """Chuyển batch index thành số samples đã thấy."""
    return np.arange(len(history)) * BATCH_SIZE


def _add_hover(ax: plt.Axes, ylabel: str = "Value") -> None:
    """
    Thêm hover tooltip cho tất cả lines trong ax.
    Hiển thị: Batch, giá trị metric, tên line.
    Yêu cầu mplcursors (pip install mplcursors).
    """
    if not _HAS_MPLCURSORS:
        return
    cursor = mplcursors.cursor(ax.get_lines(), hover=True)

    @cursor.connect("add")
    def on_add(sel):
        label = sel.artist.get_label()
        x_val = int(sel.target[0])
        y_val = sel.target[1]
        sel.annotation.set_text(
            f"{label}\nBatch: {x_val}\n{ylabel}: {y_val:.4f}"
        )
        sel.annotation.get_bbox_patch().set(
            fc="white", alpha=0.9, boxstyle="round,pad=0.4"
        )


def _legend_outside(ax: plt.Axes, fig: plt.Figure, position: str = "right") -> None:
    """
    Đặt legend ra ngoài vùng plot.

    Args:
        position: "right"  → bên phải chart
                  "bottom" → bên dưới chart
    """
    if position == "right":
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.01, 1),
            borderaxespad=0,
            framealpha=0.9,
        )
        fig.subplots_adjust(right=0.78)
    elif position == "bottom":
        handles, labels = ax.get_legend_handles_labels()
        n = len(handles)
        ax.legend(
            handles, labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=min(n, 4),
            borderaxespad=0,
            framealpha=0.9,
        )
        fig.subplots_adjust(bottom=0.20)


# ================================================================== #
#  1. SINGLE RUN PLOTS                                                 #
# ================================================================== #

def plot_prequential_accuracy(
    histories : dict[str, list],
    title     : str = "Prequential Accuracy over Time",
    filename  : Optional[str] = None,
) -> None:
    """
    Plot prequential accuracy theo thời gian cho nhiều runs.

    Args:
        histories : {"label": [acc_batch_0, acc_batch_1, ...], ...}
        title     : tiêu đề chart
        filename  : nếu có → lưu vào RESULTS_DIR

    Example:
        plot_prequential_accuracy({
            "Static"          : metrics_static.prequential_acc_history,
            "Periodic Retrain": metrics_periodic.prequential_acc_history,
        })
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (label, history) in enumerate(histories.items()):
        x = np.arange(len(history))
        ax.plot(x, history, label=label, color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Prequential Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    _legend_outside(ax, fig)
    _add_hover(ax, ylabel="Preq Acc")
    _save_or_show(fig, filename)


def plot_rolling_error(
    histories : dict[str, list],
    title     : str = "Rolling Error Rate over Time",
    filename  : Optional[str] = None,
) -> None:
    """
    Plot rolling error rate theo thời gian.

    Args:
        histories: {"label": [error_batch_0, ...], ...}
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (label, history) in enumerate(histories.items()):
        x = np.arange(len(history))
        ax.plot(x, history, label=label, color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Rolling Error Rate")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    _legend_outside(ax, fig)
    _add_hover(ax, ylabel="Roll Err")
    _save_or_show(fig, filename)


def plot_drift_measure(
    history  : list,
    label    : str = "Drift Measure",
    title    : Optional[str] = None,
    filename : Optional[str] = None,
) -> None:
    """
    Plot drift measure theo thời gian (1 run).
    Tự động thêm threshold lines cho PSI và KL.
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    x = np.arange(len(history))
    ax.plot(x, history, color=PALETTE[0], linewidth=1.5, label=label)
    ax.fill_between(x, history, alpha=0.15, color=PALETTE[0])

    # Threshold lines theo từng drift measure
    if DRIFT_MEASURE == "psi":
        ax.axhline(0.10, color="orange", linestyle="--", alpha=0.7, label="PSI=0.10 (warning)")
        ax.axhline(0.20, color="red",    linestyle="--", alpha=0.7, label="PSI=0.20 (drift)")
    elif DRIFT_MEASURE == "kl":
        ax.axhline(0.10, color="orange", linestyle="--", alpha=0.7, label="KL=0.10 (warning)")
        ax.axhline(0.25, color="red",    linestyle="--", alpha=0.7, label="KL=0.25 (drift)")
    elif DRIFT_MEASURE == "ddm":
        ax.axhline(1.0, color="orange", linestyle="--", alpha=0.7, label="DDM Warning")
        ax.axhline(2.0, color="red",    linestyle="--", alpha=0.7, label="DDM Drift")

    ax.set_xlabel("Batch Index")
    ax.set_ylabel(f"Drift Measure ({DRIFT_MEASURE.upper()})")
    ax.set_title(title or f"Drift Measure ({DRIFT_MEASURE.upper()}) over Time")
    _legend_outside(ax, fig)
    _add_hover(ax, ylabel=DRIFT_MEASURE.upper())
    _save_or_show(fig, filename)


def plot_uncertainty(
    history  : list,
    title    : str = "Model Uncertainty over Time",
    filename : Optional[str] = None,
) -> None:
    """Plot model uncertainty theo thời gian."""
    fig, ax = plt.subplots(figsize=(12, 4))

    x = np.arange(len(history))
    ax.plot(x, history, color=PALETTE[2], linewidth=1.5)
    ax.fill_between(x, history, alpha=0.15, color=PALETTE[2])

    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Uncertainty")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    _save_or_show(fig, filename)


# ================================================================== #
#  2. DASHBOARD (nhiều metrics cùng lúc)                               #
# ================================================================== #

def plot_dashboard(
    metrics_dict : dict,
    title        : str = "Stream Metrics Dashboard",
    filename     : Optional[str] = None,
) -> None:
    """
    Dashboard 4 panels cho 1 run:
        - Prequential Accuracy
        - Rolling Error Rate
        - Drift Measure
        - Model Uncertainty

    Args:
        metrics_dict: output của StreamMetrics.summary_history() hoặc dict chứa:
            {
                "label"               : str,
                "prequential_acc"     : list,
                "rolling_error"       : list,
                "drift_measure"       : list,
                "uncertainty"         : list,
            }

    Example:
        plot_dashboard({
            "label"           : "Static Baseline",
            "prequential_acc" : metrics.prequential_acc_history,
            "rolling_error"   : metrics.rolling_error_history,
            "drift_measure"   : metrics.drift_measure_history,
            "uncertainty"     : metrics.uncertainty_history,
        })
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    label = metrics_dict.get("label", "")
    color = PALETTE[0]

    def _plot_panel(ax, history, ylabel, ylim=None):
        x = np.arange(len(history))
        ax.plot(x, history, color=color, linewidth=1.5, label=label)
        ax.set_xlabel("Batch Index")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)

    # Panel 1: Prequential Accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_panel(ax1, metrics_dict["prequential_acc"], "Prequential Accuracy", (0, 1))
    ax1.set_title("Prequential Accuracy")

    # Panel 2: Rolling Error Rate
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_panel(ax2, metrics_dict["rolling_error"], "Rolling Error Rate", (0, 1))
    ax2.set_title("Rolling Error Rate")

    # Panel 3: Drift Measure
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_panel(ax3, metrics_dict["drift_measure"], f"Drift ({DRIFT_MEASURE.upper()})")
    ax3.set_title(f"Drift Measure ({DRIFT_MEASURE.upper()})")
    if DRIFT_MEASURE == "psi":
        ax3.axhline(0.10, color="orange", linestyle="--", alpha=0.6)
        ax3.axhline(0.20, color="red",    linestyle="--", alpha=0.6)
    elif DRIFT_MEASURE == "kl":
        ax3.axhline(0.10, color="orange", linestyle="--", alpha=0.6)
        ax3.axhline(0.25, color="red",    linestyle="--", alpha=0.6)

    # Panel 4: Uncertainty
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_panel(ax4, metrics_dict["uncertainty"], "Model Uncertainty", (0, 1))
    ax4.set_title("Model Uncertainty")

    _save_or_show(fig, filename)


# ================================================================== #
#  3. BASELINE COMPARISON                                              #
# ================================================================== #

def plot_baseline_comparison(
    baselines : dict[str, dict],
    filename  : Optional[str] = None,
) -> None:
    """
    So sánh nhiều baselines trên 2 metrics chính:
    Prequential Accuracy và Rolling Error Rate.

    Args:
        baselines: {
            "Static": {
                "prequential_acc": list,
                "rolling_error"  : list,
            },
            "Periodic Retrain": {...},
            ...
        }

    Example:
        plot_baseline_comparison({
            "Static"          : {"prequential_acc": [...], "rolling_error": [...]},
            "Periodic Retrain": {"prequential_acc": [...], "rolling_error": [...]},
            "Always Retrain"  : {"prequential_acc": [...], "rolling_error": [...]},
        })
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Baseline Comparison", fontsize=14, fontweight="bold")

    for i, (label, data) in enumerate(baselines.items()):
        color = PALETTE[i % len(PALETTE)]
        x1    = np.arange(len(data["prequential_acc"]))
        x2    = np.arange(len(data["rolling_error"]))

        ax1.plot(x1, data["prequential_acc"],
                 label=label, color=color, linewidth=2)
        ax2.plot(x2, data["rolling_error"],
                 label=label, color=color, linewidth=2)

    ax1.set_xlabel("Batch Index")
    ax1.set_ylabel("Prequential Accuracy")
    ax1.set_title("Prequential Accuracy")
    ax1.set_ylim(0, 1)
    _legend_outside(ax1, fig, position="bottom")
    _add_hover(ax1, ylabel="Preq Acc")

    ax2.set_xlabel("Batch Index")
    ax2.set_ylabel("Rolling Error Rate")
    ax2.set_title("Rolling Error Rate")
    ax2.set_ylim(0, 1)
    _legend_outside(ax2, fig, position="bottom")
    _add_hover(ax2, ylabel="Roll Err")

    _save_or_show(fig, filename)


def plot_retrain_events(
    rolling_error_history : list,
    retrain_batch_indices : list[int],
    title                 : str = "Retrain Events",
    filename              : Optional[str] = None,
) -> None:
    """
    Plot rolling error + đánh dấu các thời điểm retrain.

    Args:
        rolling_error_history : list error rates
        retrain_batch_indices : list batch indices khi retrain xảy ra
        
    Example:
        plot_retrain_events(
            metrics.rolling_error_history,
            retrain_indices,
        )
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    x = np.arange(len(rolling_error_history))
    ax.plot(x, rolling_error_history,
            color=PALETTE[0], linewidth=1.5, label="Rolling Error Rate")

    # Đánh dấu retrain events
    for idx in retrain_batch_indices:
        if idx < len(rolling_error_history):
            ax.axvline(idx, color="red", linestyle="--", alpha=0.4, linewidth=1)

    # Dummy line cho legend
    ax.axvline(-1, color="red", linestyle="--", alpha=0.6,
               linewidth=1, label=f"Retrain ({len(retrain_batch_indices)} times)")

    ax.set_xlabel("Batch Index")
    ax.set_ylabel("Rolling Error Rate")
    ax.set_title(title)
    _legend_outside(ax, fig)
    _add_hover(ax, ylabel="Roll Err")
    _save_or_show(fig, filename)


# ================================================================== #
#  4. RL AGENT EVALUATION                                              #
# ================================================================== #

def plot_cumulative_reward(
    reward_histories : dict[str, list],
    title            : str = "Cumulative Reward over Episodes",
    filename         : Optional[str] = None,
) -> None:
    """
    Plot cumulative reward theo episodes cho các RL agents.

    Args:
        reward_histories: {"Q-Learning": [ep1_reward, ep2_reward, ...], ...}
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (label, rewards) in enumerate(reward_histories.items()):
        cumulative = np.cumsum(rewards)
        x          = np.arange(len(rewards))
        ax.plot(x, cumulative, label=label,
                color=PALETTE[i % len(PALETTE)], linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title(title)
    _legend_outside(ax, fig)
    _add_hover(ax, ylabel="Cum Reward")
    _save_or_show(fig, filename)


def plot_action_distribution(
    action_counts : dict[str, dict[str, int]],
    title         : str = "Action Distribution per Agent",
    filename      : Optional[str] = None,
) -> None:
    """
    Bar chart so sánh phân phối actions giữa các agents.

    Args:
        action_counts: {
            "Q-Learning": {"no_action": 300, "partial_update": 100, ...},
            "SARSA"     : {"no_action": 250, ...},
        }
    """
    action_names = ["no_action", "partial_update", "full_retrain",
                    "alert", "switch_model"]
    agents       = list(action_counts.keys())
    n_agents     = len(agents)
    n_actions    = len(action_names)

    x      = np.arange(n_actions)
    width  = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, agent in enumerate(agents):
        counts = [action_counts[agent].get(a, 0) for a in action_names]
        ax.bar(x + i * width, counts, width,
               label=agent, color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.set_xlabel("Action")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(x + width * (n_agents - 1) / 2)
    ax.set_xticklabels(action_names, rotation=15)
    _legend_outside(ax, fig)
    _save_or_show(fig, filename)


def plot_agent_vs_baselines(
    baselines : dict[str, list],
    agents    : dict[str, list],
    metric    : str = "prequential_acc",
    ylabel    : str = "Prequential Accuracy",
    title     : str = "RL Agents vs Baselines",
    filename  : Optional[str] = None,
) -> None:
    """
    So sánh RL agents với baselines trên 1 metric.

    Args:
        baselines : {"Static": [acc history], ...}  – dashed lines
        agents    : {"Q-Learning": [acc history], ...} – solid lines
        metric    : tên metric để hiển thị
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    # Baselines – dashed
    for i, (label, history) in enumerate(baselines.items()):
        x = np.arange(len(history))
        ax.plot(x, history, label=f"{label} (baseline)",
                color=PALETTE[i % len(PALETTE)],
                linestyle="--", linewidth=1.5, alpha=0.7)

    # RL Agents – solid
    offset = len(baselines)
    for i, (label, history) in enumerate(agents.items()):
        x = np.arange(len(history))
        ax.plot(x, history, label=label,
                color=PALETTE[(offset + i) % len(PALETTE)],
                linestyle="-", linewidth=2)

    ax.set_xlabel("Batch Index")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _legend_outside(ax, fig)
    _add_hover(ax, ylabel=ylabel)
    _save_or_show(fig, filename)


# ================================================================== #
#  QUICK TEST                                                          #
# ================================================================== #

if __name__ == "__main__":
    # Tạo dummy data để test các hàm plot
    n = 200
    x = np.arange(n)

    # Simulate degrading accuracy
    static_acc   = 0.82 - 0.0008 * x + np.random.normal(0, 0.01, n)
    periodic_acc = 0.80 - 0.0003 * x + np.random.normal(0, 0.01, n)
    static_acc   = np.clip(static_acc,   0, 1)
    periodic_acc = np.clip(periodic_acc, 0, 1)

    # Test plot_baseline_comparison
    plot_baseline_comparison({
        "Static"          : {
            "prequential_acc": static_acc.tolist(),
            "rolling_error"  : (1 - static_acc).tolist(),
        },
        "Periodic Retrain": {
            "prequential_acc": periodic_acc.tolist(),
            "rolling_error"  : (1 - periodic_acc).tolist(),
        },
    }, filename="test_baseline_comparison.png")

    # Test plot_dashboard
    plot_dashboard({
        "label"          : "Static Baseline",
        "prequential_acc": static_acc.tolist(),
        "rolling_error"  : (1 - static_acc).tolist(),
        "drift_measure"  : (0.05 + 0.001 * x + np.random.normal(0, 0.01, n)).tolist(),
        "uncertainty"    : (0.3  + 0.001 * x + np.random.normal(0, 0.02, n)).tolist(),
    }, filename="test_dashboard.png")

    print("Visualizer test passed!")