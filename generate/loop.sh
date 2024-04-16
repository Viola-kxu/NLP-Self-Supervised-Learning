#!/bin/bash

times=1

for i in $(seq 1 $times); do
    echo "Iteration $i"
    python question_generate.py
    python filter_questions.py
    python answer_generate.py
done
