#!/bin/bash
trap "echo Program crashed. Restarting" ERR
while true
do
    python trial_spacy3.py
    sleep 1
done
