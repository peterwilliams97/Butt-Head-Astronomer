#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f completed.e_spacy_lstmx_80a.txt ]
do
    python trial_spacy8a.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 100
done

echo done