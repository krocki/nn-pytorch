#!/bin/bash
HOSTNAME=tau
python ptb-base.py --hidden=64 --seqlength=25 --fname="./logs/ptb/ptb-base-$HOSTNAME.dat" --timelimit=900
