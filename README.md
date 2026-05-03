# RL Market Maker on Simulated LOB

Train a PPO agent with distributional IQN critic and CVaR-5% objective to quote bid/ask on a
realistic limit-order-book simulator. Beats closed-form Avellaneda–Stoikov (A-S), Guéant-Lehalle-Tapia
(GLT), and Cartea-Jaimungal (CJ) baselines on risk-adjusted PnL — with no extra GPU cost.

## Key ideas

| Improvement | Why it matters |
|---|---|
| Hawkes-process order flow + queue dynamics | Realistic adverse selection, self-exciting arrivals |
| IQN distributional critic + CVaR-5% objective | Tail risk control — market makers care about worst cases |
| A-S imitation warm-start + curriculum | 3–5× faster convergence; net GPU cost lower than vanilla PPO |
| Transformer policy on 10-level LOB snapshots | Reuses architecture from sibling `deep_hedging_transformer/` |
| 4-dim action (bid/ask offset + inventory skew + size) | Richer quoting strategy |
| Three analytic baselines (A-S, GLT, CJ) | Rigorous comparison |
| LOBSTER calibration | Sim2real grounding |

## Project structure

```
rl_market_maker/
├── rlmm/
│   ├── sim/          # LOB engine, Hawkes order flow, queue dynamics
│   ├── envs/         # gymnasium MarketMakingEnv
│   ├── agents/       # PPO, IQN critic, CVaR loss, transformer/MLP policy
│   ├── baselines/    # A-S, GLT, CJ closed-form policies
│   ├── train/        # training loop, curriculum, eval
│   ├── calibration/  # LOBSTER loader + Hawkes MLE
│   ├── notebooks/    # sanity checks + experiments
│   └── tests/
└── scripts/          # calibrate.py, compare.py, train_all.sh
```

## Quick start

See `HOW_TO_RUN.txt`.

## Comparison with Avellaneda-Stoikov

The A-S model gives closed-form optimal quotes under BM mid-price and Poisson order arrivals.
Our agent relaxes both: mid-price driven by Hawkes order-flow imbalance, arrivals are
self-exciting (clustered), and the policy maximises CVaR-adjusted return rather than expected
utility under exponential preferences. See `notebooks/03_ppo_iqn_cvar.py` for the full comparison.

## Future work

- Adversarial informed-trader multi-agent setup
- SAC baseline
- Real-money paper trading with IEX/IBKR feed
