#!/bin/bash

#BSUB -J proj2
#BSUB -q c02516

#BSUB -gpu "num=1:mode=exclusive_process"

#BSUB -W 12:00
#BSUB -R "rusage[mem=10GB]"

#BSUB -o sleeper_%J.out
#BSUB -e sleeper_%J.err

#BSUB -n 4
#BSUB -R "span[hosts=1]"

#BSUB -B
#BSUB -N

source ~/IDLCVenv/bin/activate
python CNNforproposal.py
