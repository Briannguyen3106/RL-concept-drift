"""
tune_reward.py
==============
Grid search trên reward function parameters cho MC Agent.

Chạy:
    python experiments/tune_reward.py

Output:
    experiments/results/tuning_results.json   ← raw results
    experiments/results/tuning_summary.csv    ← summary table
    experiments/results/tuning_*.png          ← plots

Thời gian ước tính: 6-10 tiếng tùy máy.
"""

import sys
import os
import json
import copy
import time
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as CFG   # import module để patch trực tiếp


# ================================================================== #
#  GRID DEFINITION                                                     #
# ================================================================== #

# Các params cần tune – thêm/bớt giá trị tuỳ ý
PARAM_GRID = {
    "partial_update_cost" : [0.1, 0.2, 0.3],
    "drift_miss_penalty"  : [0.3, 0.5],
    "drift_miss_threshold": [0.35, 0.40, 0.50],
}

# Params cố định (không tune)
FIXED = {
    "n_train_episodes": 200,
    "n_eval_episodes" : 3,
    "full_retrain_cost": 0.3,
    "no_action_cost"   : 0.0,
    "alert_cost"       : 0.0,
    "switch_model_cost": 0.05,
}


# ================================================================== #
#  HELPERS                                                             #
# ================================================================== #

def patch_config(params: dict) -> None:
    """
    Patch config module trực tiếp với params hiện tại.
    Vì drift_env.py import từ config lúc runtime nên patch này có hiệu lực.
    """
    CFG.ACTION_COSTS = {
        "no_action"     : FIXED["no_action_cost"],
        "partial_update": params["partial_update_cost"],
        "full_retrain"  : FIXED["full_retrain_cost"],
        "alert"         : FIXED["alert_cost"],
        "switch_model"  : FIXED["switch_model_cost"],
    }
    CFG.DRIFT_MISS_PENALTY   = params["drift_miss_penalty"]
    CFG.DRIFT_MISS_THRESHOLD = params["drift_miss_threshold"]


def make_components():
    """Tạo fresh instances của loader, clf, metrics, env, agent."""
    from utils.data_loader import AirlinesDataLoader
    from models.LightGBM import LightGBM
    from metrics.stream_metrics import StreamMetrics
    from environment.drift_env import DriftStreamEnv
    from agents.mc_agent import MCAgent

    loader  = AirlinesDataLoader()
    clf     = LightGBM()
    metrics = StreamMetrics()
    env     = DriftStreamEnv(loader, clf, metrics)
    agent   = MCAgent()

    return loader, clf, metrics, env, agent


def train_agent(env, agent, n_episodes: int) -> dict:
    """Train MC Agent và trả về training history."""
    return agent.train(env, n_episodes=n_episodes, verbose=False)


def warmup_clf(clf, loader):
    """Warm-up clf trên FULL_WINDOW_BATCHES batches cuối train stream."""
    from config import FULL_WINDOW_BATCHES, CATEGORICAL_COLS
    X_warm, y_warm = loader.get_warmup_data(n_batches=FULL_WINDOW_BATCHES)
    clf.full_retrain(X_warm, y_warm)
    return X_warm, y_warm


def evaluate_agent(agent, env, loader, n_episodes: int) -> dict:
    """
    Prequential evaluation trên test stream.
    Trả về dict metrics tổng hợp qua n_episodes.
    """
    all_preq_acc    = []
    all_rolling_err = []
    all_rewards     = []
    all_retrain_ep  = []
    all_actions     = []

    for ep in range(n_episodes):
        state     = env.reset(start_batch=0, mode="test")
        ep_reward = 0.0
        ep_retrains = 0

        while not env.is_done:
            action, pi = agent.select_action(state)
            next_state, reward, done, info = env.step(
                action, pi=pi, update_explorer=False
            )
            if done and "action_taken" not in info:
                break

            ep_reward += reward
            all_actions.append(info["action_taken"])
            if info["action_taken"] in ["partial_update", "full_retrain"]:
                ep_retrains += 1

            state = next_state

        all_rewards.append(ep_reward)
        all_retrain_ep.append(ep_retrains)
        all_preq_acc.append(env.metrics.prequential_acc_history.copy())
        all_rolling_err.append(env.metrics.rolling_error_history.copy())

    # Average qua episodes
    min_len         = min(len(h) for h in all_preq_acc)
    avg_preq_acc    = float(np.mean([h[min_len-1] for h in all_preq_acc]))
    avg_rolling_err = float(np.mean([h[min_len-1] for h in all_rolling_err]))

    action_dist = {}
    for a in all_actions:
        action_dist[a] = action_dist.get(a, 0) + 1

    # Normalize action_dist về per-episode
    for k in action_dist:
        action_dist[k] = round(action_dist[k] / n_episodes, 1)

    return {
        "final_preq_acc"    : avg_preq_acc,
        "final_rolling_err" : avg_rolling_err,
        "avg_reward"        : float(np.mean(all_rewards)),
        "std_reward"        : float(np.std(all_rewards)),
        "avg_retrains"      : float(np.mean(all_retrain_ep)),
        "action_dist"       : action_dist,
        # Policy diversity: entropy của action distribution
        "policy_entropy"    : _entropy(action_dist),
    }


def _entropy(action_dist: dict) -> float:
    """
    Shannon entropy của action distribution.
    Cao = diverse (agent dùng nhiều actions)
    Thấp = collapsed (agent chỉ dùng 1-2 actions)
    Max với 5 actions = log2(5) ≈ 2.32 bits
    """
    total = sum(action_dist.values())
    if total == 0:
        return 0.0
    probs = [v / total for v in action_dist.values() if v > 0]
    return float(-sum(p * np.log2(p) for p in probs))


def run_baselines(loader) -> dict:
    """
    Chạy baselines 1 lần (không phụ thuộc params) để so sánh.
    Chỉ cần chạy 1 lần duy nhất, không lặp qua grid.
    """
    from models.LightGBM import LightGBM
    from metrics.stream_metrics import StreamMetrics
    from config import FULL_WINDOW_BATCHES, CATEGORICAL_COLS
    from collections import deque

    X_train, y_train = loader.get_initial_train_data()
    X_warm,  y_warm  = loader.get_warmup_data(n_batches=FULL_WINDOW_BATCHES)

    results = {}

    for strategy in ["static", "periodic", "always"]:
        clf           = LightGBM()
        clf.train(X_train, y_train)
        clf.full_retrain(X_warm, y_warm)   # warm-up

        metrics       = StreamMetrics()
        window_buffer = deque(maxlen=FULL_WINDOW_BATCHES)
        retrain_count = 0
        batch_count   = 0

        # Set reference từ warm-up
        fold_size   = len(X_warm) // 5
        init_errors = [
            clf.get_error_rate(
                X_warm.iloc[i*fold_size:(i+1)*fold_size],
                y_warm.iloc[i*fold_size:(i+1)*fold_size]
            ) for i in range(5)
        ]
        metrics.set_reference(init_errors)

        for X_batch, y_batch, idx in loader.stream_test_batches():
            window_buffer.append((X_batch, y_batch))

            if strategy == "periodic" and batch_count > 0 and batch_count % 50 == 0:
                recent = list(window_buffer)
                X_w = pd.concat([b[0] for b in recent], ignore_index=True)
                y_w = pd.concat([b[1] for b in recent], ignore_index=True)
                for col in CATEGORICAL_COLS:
                    X_w[col] = X_w[col].astype("category")
                clf.full_retrain(X_w, y_w)
                retrain_count += 1

            elif strategy == "always":
                recent = list(window_buffer)
                X_w = pd.concat([b[0] for b in recent], ignore_index=True)
                y_w = pd.concat([b[1] for b in recent], ignore_index=True)
                for col in CATEGORICAL_COLS:
                    X_w[col] = X_w[col].astype("category")
                clf.full_retrain(X_w, y_w)
                retrain_count += 1

            preds      = clf.predict(X_batch)
            y_proba    = clf.predict_proba(X_batch)
            error_rate = clf.get_error_rate(X_batch, y_batch)
            metrics.update(y_batch.values, preds, error_rate, y_proba)
            batch_count += 1

        results[strategy] = {
            "final_preq_acc"    : metrics.prequential_acc_history[-1],
            "final_rolling_err" : metrics.rolling_error_history[-1],
            "avg_retrains"      : retrain_count,
        }
        print(f"  [{strategy}] preq_acc={results[strategy]['final_preq_acc']:.4f} | "
              f"retrains={retrain_count}")

    return results


# ================================================================== #
#  SCORING                                                             #
# ================================================================== #

def compute_score(eval_result: dict, baseline_always: dict) -> float:
    """
    Composite score để rank configs.

    Score = accuracy_gain - retrain_penalty + diversity_bonus

    Trong đó:
        accuracy_gain   : preq_acc so với Always baseline (normalized)
        retrain_penalty : retrains nhiều hơn Always bị phạt
        diversity_bonus : entropy của policy (agent học diverse actions)

    Weights có thể điều chỉnh tuỳ mục tiêu nghiên cứu.
    """
    W_ACC       = 0.5   # accuracy quan trọng nhất
    W_RETRAIN   = 0.3   # efficiency (ít retrain hơn Always)
    W_DIVERSITY = 0.2   # policy diversity

    always_acc      = baseline_always["final_preq_acc"]
    always_retrains = baseline_always["avg_retrains"]

    # Accuracy gain so với Always (có thể âm nếu tệ hơn)
    acc_gain = eval_result["final_preq_acc"] - always_acc

    # Retrain efficiency: >0 nếu ít hơn Always, <0 nếu nhiều hơn
    retrain_ratio   = eval_result["avg_retrains"] / max(always_retrains, 1)
    retrain_score   = 1.0 - retrain_ratio   # +1 nếu 0 retrains, 0 nếu bằng Always

    # Diversity: normalize về [0, 1] với max entropy = log2(5)
    max_entropy     = np.log2(5)
    diversity_score = eval_result["policy_entropy"] / max_entropy

    score = (W_ACC * acc_gain +
             W_RETRAIN * retrain_score +
             W_DIVERSITY * diversity_score)

    return float(score)


# ================================================================== #
#  MAIN TUNING LOOP                                                    #
# ================================================================== #

def main():
    print("=" * 70)
    print("MC Agent – Reward Function Tuning")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Tạo tất cả combinations
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    n_configs = len(combos)
    print(f"\nTotal configs: {n_configs}")
    print(f"Params: {keys}")
    for i, combo in enumerate(combos):
        print(f"  Config {i+1:>2}: {dict(zip(keys, combo))}")

    # ── Chạy baselines 1 lần ────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("Running baselines (1 time only)...")
    from utils.data_loader import AirlinesDataLoader
    loader = AirlinesDataLoader()
    baseline_results = run_baselines(loader)
    print(f"\nBaseline results:")
    for name, res in baseline_results.items():
        print(f"  {name:<10} preq_acc={res['final_preq_acc']:.4f} | "
              f"roll_err={res['final_rolling_err']:.4f} | "
              f"retrains={res['avg_retrains']}")

    # ── Grid search ─────────────────────────────────────────────────
    all_results = []
    t_total_start = time.time()

    for config_idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        print(f"\n{'═'*70}")
        print(f"Config {config_idx+1}/{n_configs}: {params}")
        print(f"{'═'*70}")

        t_start = time.time()

        # Patch config
        patch_config(params)

        # Fresh components
        loader, clf, metrics, env, agent = make_components()

        # Phase 1: Train
        print(f"[Phase 1] Training {FIXED['n_train_episodes']} episodes...")
        train_agent(env, agent, n_episodes=FIXED["n_train_episodes"])
        print(f"  States visited: {len(agent.Q)} | Temp: {agent.temp:.4f}")

        # Warm-up clf (không warm-up agent)
        print("[Warm-up] Retraining clf on last FULL_WINDOW_BATCHES batches...")
        warmup_clf(env.clf, loader)

        # Phase 2: Evaluate
        print(f"[Phase 2] Evaluating {FIXED['n_eval_episodes']} episodes on test stream...")
        eval_result = evaluate_agent(
            agent, env, loader, n_episodes=FIXED["n_eval_episodes"]
        )

        # Compute score
        score = compute_score(eval_result, baseline_results["always"])

        elapsed = time.time() - t_start
        elapsed_total = time.time() - t_total_start
        eta = (elapsed_total / (config_idx + 1)) * (n_configs - config_idx - 1)

        print(f"\n  Results:")
        print(f"    preq_acc     : {eval_result['final_preq_acc']:.4f}  "
              f"(Always: {baseline_results['always']['final_preq_acc']:.4f})")
        print(f"    rolling_err  : {eval_result['final_rolling_err']:.4f}")
        print(f"    avg_retrains : {eval_result['avg_retrains']:.1f}  "
              f"(Always: {baseline_results['always']['avg_retrains']})")
        print(f"    avg_reward   : {eval_result['avg_reward']:.2f} ± {eval_result['std_reward']:.2f}")
        print(f"    policy_entropy: {eval_result['policy_entropy']:.3f} / {np.log2(5):.3f}")
        print(f"    action_dist  : {eval_result['action_dist']}")
        print(f"    SCORE        : {score:.4f}")
        print(f"  Time: {elapsed/60:.1f} min | ETA: {eta/60:.1f} min")

        # Lưu kết quả
        record = {
            "config_idx"    : config_idx + 1,
            "params"        : params,
            "eval"          : eval_result,
            "score"         : score,
            "elapsed_sec"   : elapsed,
            "baselines"     : baseline_results,
        }
        all_results.append(record)

        # Save sau mỗi config (phòng crash)
        _save_results(all_results)

    # ── Final summary ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TUNING COMPLETE")
    print(f"Total time: {(time.time() - t_total_start)/3600:.2f} hours")
    print(f"{'='*70}")

    _print_summary(all_results, baseline_results)
    _save_csv(all_results, baseline_results)
    _plot_results(all_results, baseline_results)


# ================================================================== #
#  SAVE / PRINT                                                        #
# ================================================================== #

def _save_results(results: list) -> None:
    """Lưu raw results ra JSON sau mỗi config."""
    path = os.path.join(CFG.RESULTS_DIR, "tuning_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _save_csv(results: list, baselines: dict) -> None:
    """Lưu summary table ra CSV."""
    rows = []
    for r in results:
        row = {
            "config_idx"          : r["config_idx"],
            "partial_update_cost" : r["params"]["partial_update_cost"],
            "drift_miss_penalty"  : r["params"]["drift_miss_penalty"],
            "drift_miss_threshold": r["params"]["drift_miss_threshold"],
            "preq_acc"            : r["eval"]["final_preq_acc"],
            "rolling_err"         : r["eval"]["final_rolling_err"],
            "avg_retrains"        : r["eval"]["avg_retrains"],
            "avg_reward"          : r["eval"]["avg_reward"],
            "policy_entropy"      : r["eval"]["policy_entropy"],
            "score"               : r["score"],
        }
        # Action counts
        for action in ["no_action", "partial_update", "full_retrain", "alert", "switch_model"]:
            row[f"act_{action}"] = r["eval"]["action_dist"].get(action, 0)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    path = os.path.join(CFG.RESULTS_DIR, "tuning_summary.csv")
    df.to_csv(path, index=False)
    print(f"\n[Saved] CSV → {path}")


def _print_summary(results: list, baselines: dict) -> None:
    """In bảng summary ra console, sort theo score."""
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    print(f"\n{'─'*90}")
    print(f"{'Rank':<5} {'pu_cost':<9} {'penalty':<9} {'thresh':<8} "
          f"{'PreqAcc':<9} {'RollErr':<9} {'Retrains':<10} {'Entropy':<9} {'Score':<8}")
    print(f"{'─'*90}")

    for rank, r in enumerate(sorted_results, 1):
        p = r["params"]
        e = r["eval"]
        marker = " ← BEST" if rank == 1 else ""
        print(f"{rank:<5} {p['partial_update_cost']:<9} {p['drift_miss_penalty']:<9} "
              f"{p['drift_miss_threshold']:<8} {e['final_preq_acc']:<9.4f} "
              f"{e['final_rolling_err']:<9.4f} {e['avg_retrains']:<10.1f} "
              f"{e['policy_entropy']:<9.3f} {r['score']:<8.4f}{marker}")

    print(f"{'─'*90}")
    print(f"\nBaselines:")
    for name, res in baselines.items():
        print(f"  {name:<10} preq_acc={res['final_preq_acc']:.4f} | "
              f"retrains={res['avg_retrains']}")


def _plot_results(results: list, baselines: dict) -> None:
    """Plot comparison charts."""
    import matplotlib.pyplot as plt

    configs      = [f"C{r['config_idx']}" for r in results]
    preq_accs    = [r["eval"]["final_preq_acc"]    for r in results]
    retrains     = [r["eval"]["avg_retrains"]       for r in results]
    entropies    = [r["eval"]["policy_entropy"]     for r in results]
    scores       = [r["score"]                      for r in results]

    always_acc     = baselines["always"]["final_preq_acc"]
    always_retrain = baselines["always"]["avg_retrains"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reward Function Tuning Results", fontsize=14, fontweight="bold")

    colors = ["#4CAF50" if s == max(scores) else "#2196F3" for s in scores]

    # Panel 1: Preq Accuracy
    ax = axes[0, 0]
    ax.bar(configs, preq_accs, color=colors, alpha=0.8)
    ax.axhline(always_acc, color="orange", linestyle="--",
               linewidth=1.5, label=f"Always ({always_acc:.4f})")
    ax.set_title("Final Prequential Accuracy")
    ax.set_ylabel("Preq Acc")
    ax.legend()
    ax.set_ylim(min(preq_accs) - 0.02, max(preq_accs) + 0.02)
    for i, v in enumerate(preq_accs):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)

    # Panel 2: Avg Retrains
    ax = axes[0, 1]
    ax.bar(configs, retrains, color=colors, alpha=0.8)
    ax.axhline(always_retrain, color="orange", linestyle="--",
               linewidth=1.5, label=f"Always ({always_retrain})")
    ax.set_title("Avg Retrains per Episode")
    ax.set_ylabel("Retrains")
    ax.legend()
    for i, v in enumerate(retrains):
        ax.text(i, v + 0.5, f"{v:.0f}", ha="center", fontsize=8)

    # Panel 3: Policy Entropy
    ax = axes[1, 0]
    ax.bar(configs, entropies, color=colors, alpha=0.8)
    ax.axhline(np.log2(5), color="red", linestyle="--",
               linewidth=1.5, label=f"Max entropy ({np.log2(5):.2f})")
    ax.set_title("Policy Entropy (Diversity)")
    ax.set_ylabel("Entropy (bits)")
    ax.legend()
    for i, v in enumerate(entropies):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    # Panel 4: Composite Score
    ax = axes[1, 1]
    ax.bar(configs, scores, color=colors, alpha=0.8)
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax.set_title("Composite Score (higher = better)")
    ax.set_ylabel("Score")
    for i, v in enumerate(scores):
        ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(CFG.RESULTS_DIR, "tuning_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=120)
    print(f"[Saved] Plot → {path}")
    plt.show()


# ================================================================== #
#  ENTRY POINT                                                         #
# ================================================================== #

if __name__ == "__main__":
    main()
