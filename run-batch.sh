#!/bin/bash
HOSTNAME=macbook
EXEC=ptb-batch.py
PREFIX="./logs/ptb/ptb-batch_v1"

python ptb-base.py --hidden=64 --seqlength=25 --fname="./logs/ptb/ptb-base-$HOSTNAME.dat" --timelimit=900

for I in 1
do
	for B in 1 2 4 8 16 32 64 100
	do
	    for S in 25
		do
		    for H in 64
			do
				FNAME="$PREFIX-B$B-H$H-S$S-lr01-$HOSTNAME.dat"
				python ptb-batch.py --batchsize=$B --hidden=$H --seqlength=$S --fname=$FNAME --timelimit=900
			done
		done
	done
done
