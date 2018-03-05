#!/bin/bash
trap "echo Program crashed. Restarting" ERR


while [ ! -f ccompleted.spacy_lstmx_100.txt ]
do
    python trial_spacy10.py
    echo sleeping 1
    sleep 5
    echo sleeping 2
    sleep 5
done

echo done
