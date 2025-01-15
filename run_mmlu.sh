#!/bin/bash

# Run the Python script 10 times
for i in {1..5}
do
   python run.py --task_name mmlu_hf --update_freq 100 --window_size 500
done