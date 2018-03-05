#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f completed.spacy_lstmx_110.txt ]
do
    python trial_spacy11_submit.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 100
done

echo done
