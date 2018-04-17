#!/bin/bash

TOXIC_LOGS=/Users/pcadmin/code/Butt-Head-Astronomer/toxic/logs

mkdir $TOXIC_LOGS

for i in `seq 2 4`;
        do
                echo Gathering logs $i
                mkdir $TOXIC_LOGS/gpu$i
                cd $TOXIC_LOGS/gpu$i
                pwd
                gcloud compute scp "gpu$i:/home/pcadmin/code/Butt-Head-Astronomer/toxic/logs/*.log" .
        done

echo done
