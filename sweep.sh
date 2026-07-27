#!/bin/bash

for num_compare in 3 5 7; do #3 5 7 9 15 32
    for method in "ot"; do # "sorting_network" "neuralsort" "softsort" "ot"
        for softness in 0.1; do
            sbatch run.sh uv run docs/examples/mnist.py \
            --lr=0.001 \
            --num_steps=100000 \
            --mode=smooth \
            --task=sort \
            --num_compare=$num_compare --method=$method --softness=$softness \
            --results_csv=docs/examples/mnist/results.csv\
            --curves_csv=docs/examples/mnist/curves.csv
        done
    done
done
