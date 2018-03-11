#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f completed.p_trial_pipe_004.txt ]
do
    python trial_pipe_random.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 100
done

echo done
