#!/bin/bash

TOXIC_LOGS=/Users/pcadmin/code/Butt-Head-Astronomer/toxic/logs

echo Gathering logs instance-5
cd $TOXIC_LOGS/instance5
pwd
gcloud compute scp "instance-5:/home/pcadmin/code/Butt-Head-Astronomer/toxic/logs/*" .


for i in `seq 2 7`;
        do
                echo Gathering logs $i
                cd $TOXIC_LOGS/gpu$i
                pwd
                gcloud compute scp "gpu$i:/home/pcadmin/code/Butt-Head-Astronomer/toxic/logs/e*" .
        done

echo done
