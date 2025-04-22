#!/bin/bash

# Loading the required module
source /etc/profile
module load anaconda/2023b 

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE

# Run the script
python configs_distance.py $LLSUB_RANK $LLSUB_SIZE