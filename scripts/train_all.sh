#!/bin/bash
# Train all ablation variants and save to checkpoints/

set -e

echo "=== RL Market Maker — Full Training Run ==="
echo "Device: auto (H100 if available)"

# Ablation 1: MLP + scalar critic + mean objective (vanilla PPO baseline)
echo ""
echo "[1/4] MLP + Scalar + Mean (vanilla PPO baseline)"
python -m rlmm.train.train \
    --policy mlp \
    --critic scalar \
    --objective mean \
    --warm-start none \
    --curriculum off \
    --n-envs 16 \
    --total-steps 1000000 \
    --save-dir checkpoints/

# Ablation 2: MLP + IQN + CVaR
echo ""
echo "[2/4] MLP + IQN + CVaR"
python -m rlmm.train.train \
    --policy mlp \
    --critic iqn \
    --objective cvar \
    --warm-start none \
    --curriculum off \
    --n-envs 16 \
    --total-steps 1000000 \
    --save-dir checkpoints/

# Ablation 3: Transformer + IQN + CVaR (no warm-start, no curriculum)
echo ""
echo "[3/4] Transformer + IQN + CVaR (no warm-start)"
python -m rlmm.train.train \
    --policy transformer \
    --critic iqn \
    --objective cvar \
    --warm-start none \
    --curriculum off \
    --n-envs 16 \
    --total-steps 2000000 \
    --save-dir checkpoints/

# Main: Transformer + IQN + CVaR + A-S warm-start + curriculum
echo ""
echo "[4/4] Transformer + IQN + CVaR + A-S warm-start + Curriculum [MAIN]"
python -m rlmm.train.train \
    --policy transformer \
    --critic iqn \
    --objective cvar \
    --warm-start as \
    --curriculum on \
    --n-envs 16 \
    --total-steps 2000000 \
    --save-dir checkpoints/

echo ""
echo "=== All done. Run notebooks/03_ppo_iqn_cvar.py for comparison plots. ==="
