# Experiment Configurations

This directory contains various multi-agent RL experiment configurations designed to test different algorithms and hyperparameters for the Speaker-Listener task.

## Algorithms Supported

- **MATD3** (Multi-Agent Twin Delayed DDPG) - Off-policy with twin critics and delayed updates
- **MADDPG** (Multi-Agent DDPG) - Off-policy with single critic, simpler than MATD3
- **IPPO** (Independent PPO) - On-policy with stochastic policies for better exploration

## Quick Reference

| Configuration | Algorithm | Key Focus | Best For | Training Time |
|--------------|-----------|-----------|----------|---------------|
| `baseline.yaml` | MATD3 | Original settings | Baseline comparison | ~6 hours |
| `improved.yaml` | MATD3 | Balanced improvements | General performance boost | ~6 hours |
| `high_lr.yaml` | MATD3 | Fast adaptation | Quick learning | ~5 hours |
| `large_batch.yaml` | MATD3 | Stable gradients | Reducing variance | ~6 hours |
| `deep_network.yaml` | MATD3 | Feature learning | Complex environments | ~8 hours |
| `aggressive_mutation.yaml` | MATD3 | Evolutionary search | Finding optimal architecture | ~7 hours |
| `stable_learning.yaml` | MATD3 | Conservative training | Avoiding instability | ~7 hours |
| `fast_learning.yaml` | MATD3 | Rapid updates | Quick iteration | ~5 hours |
| `large_population.yaml` | MATD3 | Population diversity | Evolutionary exploration | ~9 hours |
| `maddpg_baseline.yaml` | MADDPG | Alternative algorithm | Simpler than MATD3 | ~6 hours |
| `ippo_baseline.yaml` | IPPO | On-policy learning | Better exploration | ~6 hours |

## Detailed Descriptions

### 1. baseline.yaml
**Hypothesis**: Original configuration baseline
**Key Parameters**:
- Population: 4 agents
- Batch size: 128
- LR: 0.0001 (actor), 0.001 (critic)
- Network: [64] hidden layers

**Use Case**: Reference point for all comparisons

### 2. improved.yaml
**Hypothesis**: Larger networks + higher learning rates + more exploration improves performance
**Key Parameters**:
- Population: 6 agents (↑50%)
- Batch size: 256 (↑100%)
- LR: 0.0003 (actor), 0.0015 (critic)
- Network: [128, 128] deep architecture
- Higher exploration noise: 0.15

**Use Case**: General-purpose improvement attempt

### 3. high_lr.yaml
**Hypothesis**: Faster learning rates enable quicker adaptation
**Key Parameters**:
- LR: 0.001 (actor, ↑10x), 0.005 (critic, ↑5x)
- Same architecture as baseline

**Use Case**: Test if faster learning helps in this environment
**Risk**: May be unstable or overshoot optimal policies

### 4. large_batch.yaml
**Hypothesis**: Larger batches provide more stable gradient estimates
**Key Parameters**:
- Batch size: 512 (↑4x)
- LR scaled up: 0.0003 (actor), 0.002 (critic)
- Memory: 200k (↑2x)

**Use Case**: Reduce training variance, more stable learning
**Trade-off**: Requires more memory, slightly slower updates

### 5. deep_network.yaml
**Hypothesis**: Deeper networks learn better representations
**Key Parameters**:
- Latent dim: 256 (↑4x)
- Actor: [256, 256, 128, 128] (4 layers)
- Critic: [256, 256, 128] (3 layers)
- Lower LR: 0.0001 (actor), 0.0005 (critic)
- Slower tau: 0.005

**Use Case**: Complex feature extraction from observations
**Trade-off**: Slower training, requires more memory

### 6. aggressive_mutation.yaml
**Hypothesis**: Strong evolutionary search finds better architectures
**Key Parameters**:
- Population: 6 agents
- No mutation prob: 0.05 (↓75% - almost always mutate)
- Mutation SD: 0.2 (↑2x strength)
- Wide HPO ranges
- Higher exploration noise: 0.2

**Use Case**: Let evolution discover optimal hyperparameters
**Trade-off**: May be unstable, needs longer to converge

### 7. stable_learning.yaml
**Hypothesis**: Conservative settings prevent training instability
**Key Parameters**:
- Very low LR: 0.00005 (actor), 0.0005 (critic)
- High gamma: 0.99 (long-term planning)
- Slow tau: 0.001 (very gradual updates)
- Learning delay: 5000 steps
- Low exploration: 0.05
- High no-mutation: 0.5

**Use Case**: When other configs show instability
**Trade-off**: Slower learning, may not explore enough

### 8. fast_learning.yaml
**Hypothesis**: Frequent updates enable rapid adaptation
**Key Parameters**:
- Learn step: 20 (↓80% - 5x more frequent)
- Small batches: 64
- Fast tau: 0.02
- Policy freq: 1 (every step)

**Use Case**: Quickly respond to environment changes
**Trade-off**: Higher computational cost, may be noisy

### 9. large_population.yaml
**Hypothesis**: More agents = better evolutionary diversity
**Key Parameters**:
- Population: 8 agents (↑2x)
- Lower no-mutation: 0.15
- Higher mutation SD: 0.12

**Use Case**: Strong evolutionary search with diverse strategies
**Trade-off**: 2x computational cost (8 agents vs 4)

### 10. maddpg_baseline.yaml
**Algorithm**: MADDPG (Multi-Agent DDPG)
**Hypothesis**: Simpler algorithm (single critic) may work better than MATD3
**Key Parameters**:
- Algorithm: MADDPG instead of MATD3
- No delayed policy updates (MATD3 has policy_freq=2)
- Single critic (not twin critics)
- Same hyperparameters as baseline otherwise

**Use Case**: Test if MATD3's complexity is necessary
**Advantage**: Simpler, faster per-step training
**Disadvantage**: May be less stable than MATD3's twin critics

### 11. ippo_baseline.yaml
**Algorithm**: IPPO (Independent PPO)
**Hypothesis**: On-policy learning with stochastic policies improves exploration
**Key Parameters**:
- Algorithm: IPPO instead of MATD3
- On-policy (no replay buffer)
- Stochastic policies (not deterministic)
- PPO-specific: clip_coef=0.2, ent_coef=0.01, gae_lambda=0.95

**Use Case**: Test fundamentally different learning paradigm
**Advantage**: Better exploration, more stable training
**Disadvantage**: Less sample-efficient (on-policy)

## Running Experiments

### Run a single experiment
```bash
docker compose run --rm training python main.py --config configs/experiments/high_lr.yaml
```

### Run all experiments in parallel (RECOMMENDED)
The script automatically runs up to 4 experiments in parallel since each uses ~20% GPU:

```bash
./run_all_experiments.sh
```

**Features**:
- Runs up to 4 experiments concurrently
- Automatically skips already completed experiments
- Monitors running containers and waits for slots
- ~4x faster than sequential execution

**Adjust parallelism**:
Edit `MAX_PARALLEL=4` in run_all_experiments.sh to change concurrent limit.

### Run multiple experiments sequentially
```bash
for config in baseline improved high_lr large_batch deep_network; do
  echo "Running $config..."
  docker compose run --rm training python main.py --config configs/experiments/${config}.yaml
done
```

### Run specific experiments in parallel manually
```bash
# Start experiments in background
docker compose run --rm -d --name exp1 training python main.py --config configs/experiments/maddpg_baseline.yaml
docker compose run --rm -d --name exp2 training python main.py --config configs/experiments/ippo_baseline.yaml
docker compose run --rm -d --name exp3 training python main.py --config configs/experiments/baseline.yaml

# Monitor running containers
docker ps

# Follow logs
docker logs -f exp1
```

### Monitor progress
```bash
# Watch specific experiment
tail -f progress/current_run.csv

# Summary of all experiments
python summary.py

# Real-time monitoring (update every 10 seconds)
watch -n 10 python summary.py
```

### Compare results
```bash
# List all experiments
python compare.py --list

# Compare specific experiments
python compare.py exp_20251127_143052 exp_20251127_190234
```

## Experiment Strategy

### Phase 1: Quick Tests (Run First)
1. **baseline.yaml** - Establish baseline (~6h)
2. **high_lr.yaml** - Test fast learning (~5h)
3. **fast_learning.yaml** - Test frequent updates (~5h)

**Total**: ~16 hours, 3 experiments

### Phase 2: Architecture Variations
4. **improved.yaml** - Balanced improvements (~6h)
5. **deep_network.yaml** - Deep architecture (~8h)
6. **large_batch.yaml** - Stable gradients (~6h)

**Total**: ~20 hours, 3 experiments

### Phase 3: Advanced Strategies
7. **aggressive_mutation.yaml** - Evolutionary search (~7h)
8. **stable_learning.yaml** - Conservative approach (~7h)
9. **large_population.yaml** - Population diversity (~9h)

**Total**: ~23 hours, 3 experiments

## Expected Performance

Based on research insights, these configurations are most likely to beat the baseline (-60 score):

1. **improved.yaml** ⭐⭐⭐⭐⭐ - High confidence
2. **deep_network.yaml** ⭐⭐⭐⭐ - Strong architecture
3. **aggressive_mutation.yaml** ⭐⭐⭐⭐ - Evolution finds optima
4. **large_batch.yaml** ⭐⭐⭐ - Stable learning
5. **large_population.yaml** ⭐⭐⭐ - Diversity helps

Riskier but potentially high-reward:
- **high_lr.yaml** - May be unstable or may learn very fast
- **fast_learning.yaml** - Could be noisy or could adapt quickly

## Creating Custom Experiments

To create a new experiment:

```bash
# Copy an existing config
cp configs/experiments/baseline.yaml configs/experiments/my_experiment.yaml

# Edit the file
nano configs/experiments/my_experiment.yaml

# Update name and description
# Modify hyperparameters as needed

# Run it
docker compose run --rm training python main.py --config configs/experiments/my_experiment.yaml
```

## Tips

1. **Start with baseline** - Always run baseline first to have a reference point
2. **Run overnight** - Each experiment takes 5-9 hours
3. **Monitor progress** - Use `python summary.py` to check training
4. **Compare systematically** - Compare each variant against baseline
5. **Document findings** - Note which changes helped/hurt performance
6. **Iterate** - Create new configs based on what works

## Troubleshooting

- **Training unstable**: Try `stable_learning.yaml`
- **Slow convergence**: Try `high_lr.yaml` or `fast_learning.yaml`
- **Poor exploration**: Try `improved.yaml` or `aggressive_mutation.yaml`
- **Reaching -60 but not better**: Try `deep_network.yaml` or `large_population.yaml`
