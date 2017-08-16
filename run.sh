#!/bin/bash
HOSTNAME=galileo

for I in 1 2 3 4 5 .. 100
do
    for S in 5 10 25 50
	do
	    for H in 16 32 64 100 256
		do
			FNAME="./logs/ptb/ptb-base-H$H-S$S-lr01-$HOSTNAME.dat"
			python ptb-base.py --hidden=$H --seqlength=$S --fname=$FNAME
		done
	done
done
