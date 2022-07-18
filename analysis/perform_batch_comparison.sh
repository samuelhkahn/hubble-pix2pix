#!/bin/bash
lines=$(ls results | wc -l)
lines=$((lines/3))
for (( i = 1; i <= $lines; i++ )) 
do 
	echo $i
	python3 compare-sr-fits.py -v -hst results/hst$i.fits -hsc results/hsc$i.fits -sr results/sr$i.fits 
done

# python3 compare-sr-fits.py -v -hst results/hst1.fits -hsc results/hsc1.fits -sr results/sr1.fits 
