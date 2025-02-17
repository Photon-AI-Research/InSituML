#!/usr/bin/env python3

from toy_cl import train, plot

# Run one training only
##train(use_mem=False)

for use_mem in [True, False]:
    # Run train, save "data_file" to disk, return data_file name
    data_file = train(use_mem=use_mem)

    # Load data_file from disk and plot. Comment out train() call above to play
    # only with plotting.
    plot(data_file)
