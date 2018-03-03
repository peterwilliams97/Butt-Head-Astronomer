#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f completed.spacy_lstm20.txt ]
do
    python trial_spacy5_submit.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 5
done

echo done
