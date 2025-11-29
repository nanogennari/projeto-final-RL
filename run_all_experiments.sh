#!/bin/bash
# Run all experiment configurations in parallel (skips already completed)

MAX_PARALLEL=2  # Run up to 2 experiments in parallel

echo "=========================================="
echo "Running All Experiments (Parallel Mode)"
echo "Max concurrent: $MAX_PARALLEL"
echo "=========================================="

configs=(
  "baseline"
  "improved"
  "high_lr"
  "large_batch"
  "deep_network"
  "aggressive_mutation"
  "stable_learning"
  "fast_learning"
  "large_population"
  "maddpg_baseline"
  "ippo_baseline"
)

# Function to count running containers
count_running() {
  docker ps --filter "ancestor=projeto-final-rl:latest" --format "{{.Names}}" | wc -l
}

echo ""
echo "Starting parallel experiment execution..."
echo ""

for config in "${configs[@]}"; do
  # Skip if already completed
  if [ -f "results/experiments.csv" ]; then
    # Match both old naming (config_matd3) and new naming (config_algo)
    completed=$(grep ",${config}" results/experiments.csv | grep ",completed," | tail -n 1)
    if [ -n "$completed" ]; then
      echo "  ✓ Skipping $config (already completed)"
      continue
    fi
  fi

  # Wait if we've hit the parallel limit
  while [ $(count_running) -ge $MAX_PARALLEL ]; do
    echo "  ⏳ Waiting for slot ($(count_running)/$MAX_PARALLEL running)..."
    sleep 30
  done

  # Start experiment in background
  echo "  ▶ Starting $config (slot $(expr $(count_running) + 1)/$MAX_PARALLEL)"

  # Generate unique container name
  timestamp=$(date +%s)
  container_name="exp_${config}_${timestamp}"

  docker compose run --rm -d --name "$container_name" \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    training python main.py --config configs/experiments/${config}.yaml

  sleep 5  # Brief delay to avoid race conditions
done

# Wait for all to complete
echo ""
echo "All experiments queued. Waiting for completion..."
while [ $(count_running) -gt 0 ]; do
  echo "  ⏳ $(count_running) experiments still running..."
  sleep 60
done

echo ""
echo "=========================================="
echo "✅ All experiments completed!"
echo "=========================================="
echo "View results: python compare.py --list"
