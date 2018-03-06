#!/bin/bash

TOXIC_LOGS=/Users/pcadmin/code/Butt-Head-Astronomer/toxic/logs

echo Gathering summaries instance-5
cd $TOXIC_LOGS/instance5
pwd
gcloud compute scp "instance-5:/home/pcadmin/code/Butt-Head-Astronomer/toxic/run.summaries/*" .


for i in `seq 2 6`;
        do
                echo Gathering summaries $i
                cd $TOXIC_LOGS/gpu$i
                pwd
                gcloud compute scp "gpu$i:/home/pcadmin/code/Butt-Head-Astronomer/toxic/run.summaries/*" .
        done

echo done
