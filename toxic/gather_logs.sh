#!/bin/bash

TOXIC_LOGS=/Users/pcadmin/code/Butt-Head-Astronomer/toxic/logs

mkdir $TOXIC_LOGS

# echo Gathering logs instance-5
# mkdir $TOXIC_LOGS/instance5
# cd $TOXIC_LOGS/instance5
# pwd
# gcloud compute scp "instance-5:/home/pcadmin/code/Butt-Head-Astronomer/toxic/logs/v_trial_fastt*" .


for i in `seq 2 7`;
        do
                echo Gathering logs $i
                mkdir $TOXIC_LOGS/gpu$i
                cd $TOXIC_LOGS/gpu$i
                pwd
                gcloud compute scp "gpu$i:/home/pcadmin/code/Butt-Head-Astronomer/toxic/logs/v_trial_fas*" .
        done

echo done
