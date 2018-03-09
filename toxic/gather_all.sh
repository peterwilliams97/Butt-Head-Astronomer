#!/bin/bash

TOXIC_LOGS=/Users/pcadmin/code/Butt-Head-Astronomer/toxic/logs

echo Gathering summaries and logs instance-5
cd $TOXIC_LOGS/instance5
pwd
gcloud compute scp "instance-5:/home/pcadmin/code/Butt-Head-Astronomer/toxic/run.summaries/*" .
gcloud compute scp "instance-5:/home/pcadmin/code/Butt-Head-Astronomer/toxic/logs/*" .


for i in `seq 2 7`;
        do
                echo Gathering summaries and logs $i
                cd $TOXIC_LOGS/gpu$i
                pwd
                gcloud compute scp "gpu$i:/home/pcadmin/code/Butt-Head-Astronomer/toxic/run.summaries/*" .
                gcloud compute scp "gpu$i:/home/pcadmin/code/Butt-Head-Astronomer/toxic/logs/*" .
        done

echo done
