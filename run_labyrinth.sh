#!/bin/bash

filenames=("ll_pgiql_2hid-lstm.csv" "ll_pgiql_3hid-lstm.csv" "ll_pgiql_4hid-lstm.csv" "ll_pgiql_5hid-lstm.csv")
numlatents=(2 3 4 5)

for i in "${!filenames[@]}"; do
    python -m src.train_labyrinth \
        --ll_filename "${filenames[i]}" \
        --num_repeats 3 \
        --num_latents "${numlatents[i]}" \
        > "${filenames[i]%.csv}.log" 2>&1
done