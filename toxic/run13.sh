#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f completed.spacy_lstm130_flip.txt ]
do
    python trial_spacy13.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 100
done

echo done
