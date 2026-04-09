#!/bin/bash
# Wait for fullpool to finish, then start thinking ablation
FULLPOOL_PID=$1
echo "Waiting for fullpool sweep (PID $FULLPOOL_PID) to finish..."
while kill -0 $FULLPOOL_PID 2>/dev/null; do
    sleep 30
done
echo "Fullpool complete. Starting thinking ablation..."
PYTHONUNBUFFERED=1 .venv/bin/python3 -u sweep_thinking.py > results/sweep_v3_thinking/run.log 2>&1
echo "Thinking ablation complete."
