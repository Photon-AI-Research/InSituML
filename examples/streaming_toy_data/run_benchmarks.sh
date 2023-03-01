#!/bin/bash

for BLOCKS in 2 4 6 8
do
	python scaling_benchmark.py --coupling_blocks $BLOCKS
	python scaling_benchmark.py --coupling_blocks $BLOCKS
	python scaling_benchmark.py --coupling_blocks $BLOCKS
done
