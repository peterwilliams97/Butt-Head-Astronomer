#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f completed.spacy_lstm21_flip.txt ]
do
    python trial_spacy6_flip.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 5
done

echo done
