#!/usr/bin/env python3
"""
Publication-quality training plots for Hausarbeit.
All labels in English.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})

COLORS = {
    "reward":  "#2196F3",
    "success": "#4CAF50",
    "grasped": "#FF9800",
    "on_belt": "#9C27B0",
    "ev":      "#F44336",
    "vloss":   "#009688",
    "raw":     "#BBDEFB",
    "ep_len":  "#607D8B",
}

LOG_PATH = "logs/stable_training_realistic_cube_physics_1/events.out.tfevents.1774554981.PaulsLaptop.7724.0"
OUT_DIR = "plots_hausarbeit"
os.makedirs(OUT_DIR, exist_ok=True)


def load(ea, tag):
    items = ea.scalars.Items(tag)
    steps = np.array([e.step for e in items])
    vals = np.array([e.value for e in items])
    return steps, vals


def smooth(vals, window=30):
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    padded = np.pad(vals, (window // 2, window - window // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def fmt_steps(x, pos):
    if x >= 1e6:
        return f"{x/1e6:.1f}M"
    elif x >= 1e3:
        return f"{x/1e3:.0f}k"
    return str(int(x))


# ── Load data ─────────────────────────────────────────────────────────────────
ea = EventAccumulator(LOG_PATH)
ea.Reload()

s_rew,    v_rew    = load(ea, "rollout/ep_rew_mean")
s_succ,   v_succ   = load(ea, "rollout/success_rate")
s_grasp,  v_grasp  = load(ea, "rollout/reached_grasped_rate")
s_belt,   v_belt   = load(ea, "rollout/reached_on_belt_rate")
s_gratio, v_gratio = load(ea, "rollout/grasped_ratio_mean")
s_bratio, v_bratio = load(ea, "rollout/on_belt_ratio_mean")
s_ev,     v_ev     = load(ea, "train/explained_variance")
s_vl,     v_vl     = load(ea, "train/value_loss")
s_ep,     v_ep     = load(ea, "rollout/ep_len_mean")
s_lr,     v_lr     = load(ea, "train/learning_rate")
s_ec,     v_ec     = load(ea, "train/ent_coef")
s_kl,     v_kl     = load(ea, "train/approx_kl")
s_cf,     v_cf     = load(ea, "train/clip_fraction")
s_std,    v_std    = load(ea, "train/std")
s_pgl,    v_pgl    = load(ea, "train/policy_gradient_loss")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Main training overview (2×2)
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
fig.subplots_adjust(hspace=0.44, wspace=0.32)

# ── (a) Episode reward ────────────────────────────────────────────────────────
ax = axes[0, 0]
ax.plot(s_rew, v_rew, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
ax.plot(s_rew, smooth(v_rew, 40), color=COLORS["reward"], linewidth=2.0, label="smoothed")
ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
ax.set_title("(a) Mean Episode Return")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Return")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="lower right")

# ── (b) Success rate ──────────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(s_succ, v_succ * 100, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
sm_succ = smooth(v_succ, 40) * 100
ax.plot(s_succ, sm_succ, color=COLORS["success"], linewidth=2.0)
final_succ = sm_succ[-1]
ax.axhline(final_succ, color=COLORS["success"], linestyle="--", linewidth=1.0,
           alpha=0.7, label=f"final: {final_succ:.0f}%")
ax.set_title("(b) Success Rate")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Success Rate [%]")
ax.set_ylim(-2, 102)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="lower right")

# ── (c) Milestone rates ───────────────────────────────────────────────────────
ax = axes[1, 0]
ax.plot(s_grasp, smooth(v_grasp, 40) * 100, color=COLORS["grasped"],
        linewidth=2.0, label="grasp reached")
ax.plot(s_belt, smooth(v_belt, 40) * 100, color=COLORS["on_belt"],
        linewidth=2.0, label="belt placement reached")
ax.set_title("(c) Milestone Rates")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Rate [%]")
ax.set_ylim(-2, 102)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="lower right")

# ── (d) Episode length ────────────────────────────────────────────────────────
ax = axes[1, 1]
ax.plot(s_ep, v_ep, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
ax.plot(s_ep, smooth(v_ep, 40), color=COLORS["ep_len"], linewidth=2.0)
ax.axhline(500, color="gray", linestyle=":", linewidth=1.0, label="max steps (500)")
ax.set_title("(d) Mean Episode Length")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Steps")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="upper right")

fig.savefig(f"{OUT_DIR}/training_overview.pdf", bbox_inches="tight")
fig.savefig(f"{OUT_DIR}/training_overview.png", bbox_inches="tight")
print("Saved: training_overview.pdf/png")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1b – Return & Success Rate side by side (1×2)
# ═══════════════════════════════════════════════════════════════════════════════
fig_rs, axes_rs = plt.subplots(1, 2, figsize=(7.0, 2.8))
fig_rs.subplots_adjust(wspace=0.35)

ax = axes_rs[0]
ax.plot(s_rew, v_rew, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
ax.plot(s_rew, smooth(v_rew, 40), color=COLORS["reward"], linewidth=2.0, label="smoothed")
ax.axhline(0, color="gray", linestyle=":", linewidth=1.0)
ax.set_title("(a) Mean Episode Return")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Return")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="lower right")

ax = axes_rs[1]
ax.plot(s_succ, v_succ * 100, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
sm_succ = smooth(v_succ, 40) * 100
ax.plot(s_succ, sm_succ, color=COLORS["success"], linewidth=2.0)
final_succ = sm_succ[-1]
ax.axhline(final_succ, color=COLORS["success"], linestyle="--", linewidth=1.0,
           alpha=0.7, label=f"final: {final_succ:.0f}%")
ax.set_title("(b) Success Rate")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Success Rate [%]")
ax.set_ylim(-2, 102)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="lower right")

fig_rs.savefig(f"{OUT_DIR}/return_and_success.pdf", bbox_inches="tight")
fig_rs.savefig(f"{OUT_DIR}/return_and_success.png", bbox_inches="tight")
print("Saved: return_and_success.pdf/png")
plt.close(fig_rs)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE – Mean Episode Return (standalone)
# ═══════════════════════════════════════════════════════════════════════════════
fig_r, ax_r = plt.subplots(figsize=(5.5, 2.8))

ax_r.plot(s_rew, v_rew, color=COLORS["reward"], linewidth=1.2)
ax_r.axhline(0, color="gray", linestyle=":", linewidth=1.0)
ax_r.set_xlabel("Environment Steps")
ax_r.set_ylabel("Mean Episode Return")
ax_r.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))

fig_r.savefig(f"{OUT_DIR}/mean_episode_return.pdf", bbox_inches="tight")
fig_r.savefig(f"{OUT_DIR}/mean_episode_return.png", bbox_inches="tight")
print("Saved: mean_episode_return.pdf/png")
plt.close(fig_r)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Task quality: time fractions (1×2)
# Shows how much of each episode the agent spends grasping / cube on belt
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 2, figsize=(7.0, 2.8))
fig2.subplots_adjust(wspace=0.35)

ax = axes2[0]
ax.plot(s_gratio, v_gratio * 100, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
ax.plot(s_gratio, smooth(v_gratio, 40) * 100, color=COLORS["grasped"], linewidth=2.0)
ax.set_title("(a) Grasp Time Fraction")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Fraction of Episode [%]")
ax.set_ylim(-2, 55)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))

ax = axes2[1]
ax.plot(s_bratio, v_bratio * 100, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
ax.plot(s_bratio, smooth(v_bratio, 40) * 100, color=COLORS["on_belt"], linewidth=2.0)
ax.set_title("(b) Belt Time Fraction")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Fraction of Episode [%]")
ax.set_ylim(-2, 55)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))

fig2.savefig(f"{OUT_DIR}/task_quality.pdf", bbox_inches="tight")
fig2.savefig(f"{OUT_DIR}/task_quality.png", bbox_inches="tight")
print("Saved: task_quality.pdf/png")
plt.close(fig2)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – PPO diagnostics (2×2)
# ═══════════════════════════════════════════════════════════════════════════════
fig3, axes3 = plt.subplots(2, 2, figsize=(7.0, 5.0))
fig3.subplots_adjust(hspace=0.44, wspace=0.32)

# ── (a) Linear schedules — normalised to [0,1] to avoid dual-axis clutter ─────
ax = axes3[0, 0]
lr_norm = (v_lr - v_lr.min()) / (v_lr.max() - v_lr.min())
ec_norm = (v_ec - v_ec.min()) / (v_ec.max() - v_ec.min())
ax.plot(s_lr, lr_norm, color="#1565C0", linewidth=2.0,
        label=r"learning rate ($3{\times}10^{-4}\!\to\!3{\times}10^{-5}$)")
ax.plot(s_ec, ec_norm, color="#E65100", linewidth=2.0, linestyle="--",
        label=r"entropy coef. ($10^{-2}\!\to\!3{\times}10^{-3}$)")
ax.set_title("(a) Annealed Schedules")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Normalised Value")
ax.set_ylim(-0.05, 1.15)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="upper right", fontsize=7.5)

# ── (b) Approx. KL divergence ─────────────────────────────────────────────────
ax = axes3[0, 1]
ax.plot(s_kl, v_kl, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
ax.plot(s_kl, smooth(v_kl, 40), color="#1976D2", linewidth=2.0)
ax.axhline(0.01, color="gray", linestyle=":", linewidth=1.0, label="target (0.01)")
ax.set_title("(b) Approx. KL Divergence")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("KL Divergence")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="upper right")

# ── (c) Explained variance ────────────────────────────────────────────────────
ax = axes3[1, 0]
ax.plot(s_ev, v_ev, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
ax.plot(s_ev, smooth(v_ev, 40), color=COLORS["ev"], linewidth=2.0)
ax.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, label="ideal (1.0)")
ax.axhline(0.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.4)
ax.set_title("(c) Explained Variance")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Explained Variance")
ax.set_ylim(-0.1, 1.1)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax.legend(loc="lower right")

# ── (d) Policy std (SDE) ──────────────────────────────────────────────────────
ax = axes3[1, 1]
ax.plot(s_std, v_std, color=COLORS["raw"], linewidth=0.6, alpha=0.5)
ax.plot(s_std, smooth(v_std, 40), color="#283593", linewidth=2.0)
ax.set_title("(d) Policy Std Dev (SDE)")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Std Dev")
ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))

fig3.savefig(f"{OUT_DIR}/ppo_diagnostics.pdf", bbox_inches="tight")
fig3.savefig(f"{OUT_DIR}/ppo_diagnostics.png", bbox_inches="tight")
print("Saved: ppo_diagnostics.pdf/png")
plt.close(fig3)


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – Milestone timeline (wide, single figure for paper body)
# ═══════════════════════════════════════════════════════════════════════════════
fig4, ax4 = plt.subplots(figsize=(6.5, 3.0))

ax4.plot(s_grasp, smooth(v_grasp, 40) * 100, color=COLORS["grasped"],
         linewidth=2.0, label="grasp reached")
ax4.plot(s_belt,  smooth(v_belt,  40) * 100, color=COLORS["on_belt"],
         linewidth=2.0, label="belt placement reached")
ax4.plot(s_succ,  smooth(v_succ,  40) * 100, color=COLORS["success"],
         linewidth=2.0, label="success (cube at belt end)")
ax4.set_xlabel("Environment Steps")
ax4.set_ylabel("Rate [%]")
ax4.set_ylim(-2, 102)
ax4.set_title("Milestone Success Rates over Training")
ax4.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_steps))
ax4.legend(loc="upper left")

fig4.savefig(f"{OUT_DIR}/milestone_rates.pdf", bbox_inches="tight")
fig4.savefig(f"{OUT_DIR}/milestone_rates.png", bbox_inches="tight")
print("Saved: milestone_rates.pdf/png")
plt.close(fig4)


# ── Summary ───────────────────────────────────────────────────────────────────
print("\nFinal values (smoothed):")
print(f"  ep_rew_mean:          {smooth(v_rew,    40)[-1]:.2f}")
print(f"  success_rate:         {smooth(v_succ,   40)[-1]*100:.1f}%")
print(f"  reached_grasped_rate: {smooth(v_grasp,  40)[-1]*100:.1f}%")
print(f"  reached_on_belt_rate: {smooth(v_belt,   40)[-1]*100:.1f}%")
print(f"  grasped_ratio_mean:   {smooth(v_gratio, 40)[-1]*100:.1f}%")
print(f"  on_belt_ratio_mean:   {smooth(v_bratio, 40)[-1]*100:.1f}%")
print(f"  ep_len_mean:          {smooth(v_ep,     40)[-1]:.1f} steps")
print(f"  explained_variance:   {smooth(v_ev,     40)[-1]:.4f}")
print(f"  total_timesteps:      {s_rew[-1]:,}")
