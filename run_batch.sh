#!/bin/bash
HOSTNAME=macbook
EXEC=ptb-batch.py
PREFIX="./logs/ptb/ptb-batch"

for I in 1 2 3 4 5 6 7 8 9 10
do
	for B in 1 2 4 8 16 32 64 100
	do
	    for S in 25
		do
		    for H in 64
			do
				FNAME="$PREFIX-B$B-H$H-S$S-lr01-$HOSTNAME.dat"
				python ptb-batch.py --batchsize=$B --hidden=$H --seqlength=$S --fname=$FNAME
			done
		done
	done
done
