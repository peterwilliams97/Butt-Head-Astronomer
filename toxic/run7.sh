#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f completed.spacy_lstm22.txt ]
do
    python trial_spacy7.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 5
done

echo done
