#!/bin/bash

mkdir -p logs

for i in {1..5}
do
    timestamp=$(date +"%Y%m%d_%H%M%S")
    logfile="logs/run_${i}_${timestamp}.log"

    echo "--------------------------------------------------"
    echo "Starting run $i (logging to $logfile)"
    echo "--------------------------------------------------"

    python run.py --task_name gsm8k | tee "$logfile"
done
