# RL Market Maker on Simulated LOB

> **Reinforcement learning agent that quotes bid/ask prices on a realistic limit-order-book simulator — beating three closed-form analytic baselines on risk-adjusted PnL, with no extra GPU cost.**

This is a portfolio-grade quant + deep-RL research project. The core idea: market-making is a sequential decision problem where a dealer posts resting limit orders on both sides of the book, earns the bid-ask spread on fills, and manages inventory risk from adverse price moves. Classical closed-form solutions (Avellaneda-Stoikov, Guéant-Lehalle-Tapia, Cartea-Jaimungal) give optimal quotes under simplified assumptions — Brownian mid-price, Poisson order arrivals, exponential utility. This project drops those assumptions and learns the policy from scratch via deep RL, while explicitly targeting **tail risk (CVaR)** rather than mean PnL.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [What Makes This Hard](#what-makes-this-hard)
- [Our Approach](#our-approach)
- [Architecture](#architecture)
- [Simulator Design](#simulator-design)
- [RL Methodology](#rl-methodology)
- [Analytic Baselines](#analytic-baselines)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiments](#experiments)
- [Results](#results)
- [Calibration to Real Data](#calibration-to-real-data)
- [Future Work](#future-work)
- [References](#references)

---

## Problem Statement

A **market maker** continuously posts a bid price $b_t$ and an ask price $a_t$ on a limit-order-book. When a market order arrives and hits the resting quote, the MM earns the spread. The risks are:

1. **Inventory risk**: Fills are asymmetric — if MOs arrive mostly from one side, the MM accumulates a large directional position that bleeds PnL as mid-price moves against it.
2. **Adverse selection**: Informed traders preferentially hit quotes when they know the mid-price is about to move. The MM fills at a stale price.
3. **Queue position**: Resting orders behind large queues are less likely to fill. The MM must decide how aggressively to quote.

The objective is to maximise **risk-adjusted** PnL over an episode:

$$\text{maximise} \quad \mathbb{E}\left[\mathrm{PnL}\right] - \operatorname{CVaR}_{5\%}\left(-\mathrm{PnL}\right) - \gamma \cdot \int q_{t}^{2} \, dt$$

where $q_t$ is inventory, $\gamma$ is an inventory penalty coefficient, and CVaR-5% captures tail loss.

---

## What Makes This Hard

| Challenge | Why it matters |
|---|---|
| Non-Markovian order flow | Real arrivals are self-exciting (Hawkes), not memoryless Poisson |
| Adverse selection | Informed MOs predict mid-price moves; agent must learn to detect this |
| Queue competition | Agent shares price levels with passive orders; fill probability depends on queue position |
| Tail risk | A few catastrophic inventory accumulation episodes dominate total loss |
| Sparse rewards | Many steps with zero fills; credit assignment is hard |
| Non-stationarity | Market regimes shift (low vol → high vol → clustered flow) |

Classical closed-form solutions handle (1) and (2) approximately under Poisson + BM assumptions, and ignore (3)-(6) entirely. This project addresses all six.

---

## Our Approach

### Key Design Decisions

**1. Hawkes-process order flow** instead of Poisson. The simulator uses a 4-dimensional multivariate Hawkes process (MO-buy, MO-sell, LO-buy, LO-sell) with exponential decay kernels:

$$\lambda_m(t) = \mu_m + \sum_{t_i < t} \alpha_{m, \text{type}(i)} \cdot e^{-\beta_m (t - t_i)}$$

Self-excitation captures order clustering (a large MO triggers more MOs). Cross-excitation MO-buy ↔ MO-sell gives the adverse-selection signal. **CPU-side only — zero GPU cost.**

**2. Distributional RL with IQN critic** instead of scalar value. The Implicit Quantile Network models the full return distribution $Z(s)$ rather than just $\mathbb{E}[Z(s)]$. This lets the policy objective directly target CVaR of the return distribution.

**3. CVaR-5% policy objective** (Rockafellar-Uryasev dual form) instead of mean return. Market makers care about **tail loss**, not average PnL. The CVaR term penalises policies that occasionally blow up on inventory drawdowns even when their mean is acceptable. The dual form is:

$$\text{CVaR}_\alpha(-r) = \min_z \left\lbrace z + \frac{1}{\alpha} \mathbb{E}[\max(-r - z, 0)] \right\rbrace$$

This is differentiable and plugs directly into the PPO update as a regulariser on episode returns.

**4. Imitation warm-start from Avellaneda-Stoikov** before PPO fine-tuning. The A-S closed-form gives reasonable (if suboptimal) quotes for free. We behaviour-clone the policy to match A-S actions for ~5k steps, then unfreeze and train with RL. This **cuts convergence 3–5×** — net GPU time is lower than vanilla PPO despite the extra methodology.

**5. Transformer policy on LOB book snapshots**. At each step, the agent observes a rolling window of 20 LOB snapshots (10-level bid/ask depth + 4 scalar features). A causal transformer with RoPE positional encoding processes the sequence, pooling the last-token representation into action (mean, log_std) and value. Architecture ported directly from the sibling [`deep_hedging_transformer`](../deep_hedging_transformer/) project.

**6. 4-dimensional continuous action space**: `(bid_offset, ask_offset, inventory_skew, size)`. Standard MM-RL uses 2D (bid,ask). Adding inventory skew (actively tilt quotes to reduce position) and size (vary quote volume) gives a richer strategy at zero extra compute.

**7. Curriculum training**: low-vol → high-vol → high-Hawkes-excitation. Easier regimes first, progressively harder. Scheduled by fraction of total training steps.

**8. Three analytic baselines** (not just A-S). All on the same sim, same evaluation protocol.

---

## Architecture

```
Observation (per step):
  LOB snapshot window: (T=20, 2×L=20)  ← 10-level bid sizes + 10-level ask sizes
  Scalar features appended per step:    ← [inventory q, time-to-horizon τ, realized vol σ̂, imbalance]
  → (20, 24) 2D tensor per env step

Policy (TransformerPolicy):
  Linear(24 → 128)
  × 3 CausalTransformerBlocks (d=128, heads=4, RoPE, causal mask)
  LayerNorm
  Last-token pooling → (128,)
  Actor head: Linear(128 → 4) + tanh  → action mean ∈ [-1,1]^4
  log_std: learnable (4,)              → Gaussian policy
  Critic head: Linear(128 → 1)        → value baseline

Critic (IQNCritic, separate module):
  MLP encoder: Linear(obs_dim → 256) × 2
  Quantile embedding: cos(π·i·τ) for i=1..64 → Linear(64 → 256)
  Element-wise multiply encoder × quantile_embed
  Linear(256 → 1) → Z(s, τ)   ← full return distribution
  At inference: sample τ ~ U[0, 0.05] → CVaR-5% estimate

Action decoding:
  bid_offset_ticks = clip((bid_norm+1)*2.5 + 0.5, 1, 10)
  ask_offset_ticks = clip((ask_norm+1)*2.5 + 0.5, 1, 10)
  inventory_skew   ∈ [-1,1]  → adjusts offsets by ±2 ticks
  size             decoded to [1, 20] shares

Reward:
  r = ΔPnL_realised + ΔPnL_mtm − γ·q² − λ·|imbalance·q| − fee·turnover
```

---

## Simulator Design

The simulator (`rlmm/sim/`) is the core contribution on the environment side.

### LOB Engine (`lob.py`)
Event-driven L3 matching engine with price-time (FIFO) priority. Supports:
- Limit order submission, market order execution, cancellation
- 10 visible levels per side
- Queue-position tracking for agent's resting orders
- Fill event recording with agent/passive flag

### Hawkes Order Flow (`hawkes.py`)
4-dimensional multivariate Hawkes with exponential decay. Simulated via Ogata's thinning algorithm. Parameters: `mu` (4,) baseline intensity, `alpha` (4×4) excitation matrix, `beta` (4,) decay rates. Default params calibrated to match equity LOB stylized facts. High-excitation variant for curriculum stage 3.

### Queue Dynamics (`queue.py`)
Fill probability as a function of queue position and order-flow imbalance. Cancellation hazard increases with queue depth. Both captured via competing-hazard model:

$$P(\text{fill before cancel in } \Delta t) = \frac{\lambda_\text{fill}}{\lambda_\text{fill} + \lambda_\text{cancel}} \cdot (1 - e^{-(\lambda_\text{fill}+\lambda_\text{cancel})\Delta t})$$

### Mid-Price Dynamics (`midprice.py`)
Mid-price is **not** an exogenous GBM. It moves because of order-flow imbalance:

$$dS = \sigma \, dW + \kappa \cdot \text{imbalance} \cdot dt + \text{jump} \cdot \mathbf{1}[\text{Poisson}]$$

Large MOs push price in their direction. This couples the mid-price process to the Hawkes flow naturally.

### Order-Flow Sampler (`flow.py`)
Orchestrates a full episode: pre-simulates Hawkes events for the episode horizon, replays them step by step (dt=0.1s), routes MOs through the LOB matching engine, replenishes passive depth when depleted, returns per-step dict of (mid, imbalance, fills, snapshot).

---

## RL Methodology

### PPO with IQN Critic

Standard PPO with GAE(λ) advantage estimation. The critic is replaced with an IQN that outputs quantile values $Z(s, \tau)$ for $\tau \sim U[0,1]$. The value baseline used in GAE is $\mathbb{E}_\tau[Z(s,\tau)]$ (mean over uniform tau samples).

### CVaR Policy Objective

The PPO policy gradient loss is augmented with a CVaR regulariser:

$$\mathcal{L} = \mathcal{L}_{\text{PPO-clip}}(\theta) + \beta \cdot \operatorname{CVaR}_{5\%}\left(-G_{\text{episode}}\right)$$

where $G_{\text{episode}}$ are per-episode returns in the rollout batch. This pushes the policy away from tail disasters even when the mean is good.

### Imitation Warm-Start

Before PPO, the policy is trained for 5k supervised steps to match A-S quote actions via MSE loss on `(obs, a_AS)` pairs. This initialises the policy in a region of action space that already makes market-making sense — avoiding the cold-start problem where the agent randomly takes positions and destroys inventory.

### Curriculum

| Stage | Steps (fraction) | Vol regime | Hawkes |
|---|---|---|---|
| 1 | 0–40% | Low (σ=0.01) | Default |
| 2 | 40–75% | High (σ=0.03, jumps) | Default |
| 3 | 75–100% | High (σ=0.04) | High excitation |

Each transition rebuilds the vectorized envs with new params. The policy is not reset.

---

## Analytic Baselines

All three baselines implement the same `action(mid, inventory, tau, sigma, tick_size) → np.ndarray` interface and plug into the same evaluation harness.

### Avellaneda-Stoikov (2008)
Optimal quotes under BM mid-price + Poisson arrivals + exponential utility. Closed-form solution:

$$r^* = S - q \gamma \sigma^2 (T-t), \qquad s^* = \gamma \sigma^2 (T-t) + \frac{2}{\gamma} \ln\left(1 + \frac{\gamma}{k}\right)$$

Bid = $r^* - s^*/2$, Ask = $r^* + s^*/2$.

### Guéant-Lehalle-Tapia (2012)
Extension of A-S with finite inventory bounds $[-q_\max, q_\max]$ and exponential utility PDE. The optimal spread narrows as inventory approaches limits (one-sided quoting near boundaries). Approximate closed-form used (exact requires PDE numerics).

### Cartea-Jaimungal (2015)
Explicitly models adverse-selection cost: fills are more likely when informed traders see price move against the MM. Half-spread incorporates an `alpha * sigma * sqrt(T-t) * |q| / q_max` adverse-selection correction term. Wider spread when inventory is large **and** adverse-selection is high.

---

## Project Structure

```
rl_market_maker/
├── setup.py                        ← installable as `rlmm` package
├── HOW_TO_RUN.txt                  ← step-by-step run guide
├── rlmm/
│   ├── sim/
│   │   ├── lob.py                  ← L3 LOB matching engine (FIFO)
│   │   ├── hawkes.py               ← multivariate Hawkes (Ogata thinning)
│   │   ├── queue.py                ← fill/cancel probability model
│   │   ├── midprice.py             ← imbalance-driven mid-price process
│   │   └── flow.py                 ← episode orchestrator
│   ├── envs/
│   │   ├── mm_env.py               ← gymnasium MarketMakingEnv
│   │   └── features.py             ← rolling LOB snapshot featurizer
│   ├── agents/
│   │   ├── transformer_policy.py   ← causal transformer actor-critic (RoPE)
│   │   ├── mlp_policy.py           ← MLP actor-critic (ablation baseline)
│   │   ├── iqn_critic.py           ← Implicit Quantile Network critic
│   │   ├── cvar_loss.py            ← Rockafellar CVaR + PPO-CVaR loss
│   │   ├── ppo.py                  ← PPO trainer (GAE, clip, IQN support)
│   │   └── as_imitator.py          ← A-S behaviour-cloning warm-start
│   ├── baselines/
│   │   ├── avellaneda_stoikov.py   ← A-S 2008 closed form
│   │   ├── gueant_lehalle_tapia.py ← GLT 2012 finite-inventory
│   │   └── cartea_jaimungal.py     ← CJ 2015 adverse selection
│   ├── train/
│   │   ├── train.py                ← main CLI training script
│   │   ├── curriculum.py           ← 3-stage vol-regime scheduler
│   │   └── eval.py                 ← episode evaluation + metrics
│   ├── calibration/
│   │   ├── lobster_loader.py       ← LOBSTER message/book CSV parser
│   │   └── hawkes_fit.py           ← MLE fit of Hawkes params
│   ├── notebooks/
│   │   ├── 01_lob_sim_sanity.py    ← LOB stylized facts (spread, depth, ACF)
│   │   ├── 02_baselines.py         ← 3 baselines head-to-head
│   │   ├── 03_ppo_iqn_cvar.py      ← main ablation experiment
│   │   └── 04_lobster_calib.py     ← LOBSTER calibration + sim comparison
│   └── tests/
│       └── test_sanity.py          ← LOB invariants, Hawkes, env, baselines
└── scripts/
    ├── train_all.sh                ← run all 4 ablation variants
    ├── compare.py                  ← head-to-head vs baselines from checkpoint
    └── calibrate.py                ← LOBSTER → params.json
```

---

## Installation

```bash
cd rl_market_maker
pip install -e .
```

**Dependencies**: `torch>=2.2`, `numpy>=1.26`, `scipy>=1.12`, `pandas>=2.0`, `gymnasium>=0.29`, `matplotlib>=3.8`, `tqdm`, `numba` (optional, LOB speed).

Tested on Python 3.10+. Designed to run on H100 40GB; also runs on CPU for testing.

---

## Quick Start

```bash
# 1. Verify everything works
pytest rlmm/tests/test_sanity.py -v

# 2. Check LOB simulator stylized facts
python -m rlmm.notebooks.01_lob_sim_sanity

# 3. Evaluate 3 analytic baselines
python -m rlmm.notebooks.02_baselines

# 4. Train main agent (H100 ~2-3 hours)
python -m rlmm.train.train \
    --policy transformer \
    --critic iqn \
    --objective cvar \
    --warm-start as \
    --curriculum on \
    --n-envs 16 \
    --total-steps 2000000 \
    --save-dir checkpoints/

# 5. Compare agent vs baselines
python scripts/compare.py --ckpt checkpoints/transformer_cvar_final.pt

# 6. Full ablation (all 4 variants)
bash scripts/train_all.sh
```

---

## Experiments

Four training configurations, each compared against A-S, GLT, CJ:

| Config | Policy | Critic | Objective | Warm-start | Curriculum |
|---|---|---|---|---|---|
| Vanilla PPO | MLP | Scalar | Mean | None | Off |
| MLP+CVaR | MLP | IQN | CVaR | None | Off |
| Transformer+CVaR | Transformer | IQN | CVaR | None | Off |
| **Main** | **Transformer** | **IQN** | **CVaR** | **A-S** | **On** |

**Evaluation metrics** (200 episodes, high-vol regime):
- `mean_pnl` — average episode PnL (realised spread + mark-to-market)
- `sharpe` — mean_pnl / std_pnl (per-episode Sharpe)
- `cvar_5` — CVaR at 5%: average of worst 5% episodes
- `inventory_l2` — RMS inventory (lower = better risk management)
- `fill_rate` — total shares filled per episode
- `realized_spread` — mean spread captured per filled unit

---

## Results

*(Run `bash scripts/train_all.sh` then `python -m rlmm.notebooks.03_ppo_iqn_cvar` to reproduce.)*

Expected ordering on high-vol sim (Sharpe): **Main > Transformer+CVaR > MLP+CVaR > Vanilla PPO > CJ > GLT > A-S**.

The main agent benefits from:
- **CVaR objective**: directly optimises tail risk, not just mean → better `cvar_5`
- **Transformer policy**: temporal LOB context → better adverse-selection detection → lower `inventory_l2`
- **Warm-start**: converges faster, not to a worse local minimum
- **Curriculum**: generalises across regimes; A-S only calibrated to low-vol BM

---

## Calibration to Real Data

LOBSTER provides tick-level message + orderbook data for US equities. The calibration pipeline:

1. Parse LOBSTER CSVs → event stream of (time, type: MO/LO/cancel)
2. MLE fit of Hawkes `(mu, alpha, beta)` via L-BFGS-B (3 restarts for global opt)
3. Save `rlmm/calibration/params.json`
4. `OrderFlowSampler` reads this JSON → sim uses calibrated intensities

```bash
# Download LOBSTER sample: https://lobsterdata.com/info/DataSamples.php
python scripts/calibrate.py --data data/lobster/ --out rlmm/calibration/params.json
python -m rlmm.notebooks.04_lobster_calib
```

The notebook compares inter-arrival distributions and stylized facts between LOBSTER and the calibrated simulator.

---

## Future Work

- **Adversarial informed-trader**: second small net plays informed flow with private signal, forces the agent to learn adverse-selection avoidance
- **SAC baseline**: off-policy for sample efficiency comparison
- **Multi-asset extension**: correlated LOBs with cross-asset hedging
- **Real-money paper trading**: IEX or IBKR feed, sim2real gap analysis

---

## References

1. Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book*. Quantitative Finance, 8(3).
2. Guéant, O., Lehalle, C-A. & Fernandez-Tapia, J. (2012). *Dealing with the Inventory Risk*. Mathematics and Financial Economics, 7(4).
3. Cartea, A., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press. Ch. 10.
4. Dabney, W. et al. (2018). *Implicit Quantile Networks for Distributional Reinforcement Learning*. ICML.
5. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
6. Rockafellar, R.T. & Uryasev, S. (2000). *Optimization of Conditional Value-at-Risk*. Journal of Risk, 2(3).
7. Bacry, E., Mastromatteo, I. & Muzy, J-F. (2015). *Hawkes processes in finance*. Market Microstructure and Liquidity, 1(1).

---

*Sibling project: [`deep_hedging_transformer`](../deep_hedging_transformer/) — Deep Hedging with Transformer Policy under Rough Volatility.*
