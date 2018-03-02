#!/bin/bash
trap "echo Program crashed. Restarting" ERR
while true
do
    python trial_spacy5_reductions.py
    sleep 1
done
