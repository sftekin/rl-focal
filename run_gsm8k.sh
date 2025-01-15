#!/bin/bash

# Run the Python script 10 times
for i in {1..5}
do
   python run.py --task_name gsm8k --update_freq 100 --window_size 500
done