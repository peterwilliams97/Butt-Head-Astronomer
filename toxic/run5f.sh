#!/bin/bash
trap "echo Program crashed. Restarting" ERR
while true
do
    python trial_spacy5_flip.py
    echo sleeping
    sleep 100
done
