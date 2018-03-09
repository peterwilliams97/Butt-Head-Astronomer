#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f completed.e_spacy_lstmx_85ar_40000.txt ]
do
    python trial_spacy8a_40000r.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 100
done

echo done
