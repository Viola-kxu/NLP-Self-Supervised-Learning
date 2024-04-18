#!/bin/bash

iter=40

for i in $(seq 1 $iter); do
    echo "Iteration $i"
    python question_generate.py
    python filter_questions.py
done
