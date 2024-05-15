import torch
import numpy as np
import data_preprocessing
import os

path_to_radiation_data = "/bigdata/hplsim/production/LWFA_radiation_new/LWFArad_data_example/LWFArad_data_example/radiationOpenPMD"
items = [
    path_to_radiation_data + "/" + next_file
    for next_file in os.listdir(path_to_radiation_data)
]
get_radiation_data = data_preprocessing.get_radiation_spectra_2_projections
chunk_size = 100000

for ind, item in enumerate(items):
    arr = get_radiation_data(ind, items, chunk_size)
    arr = arr.detach().cpu().numpy()

    if item == items[0]:
        vmin = np.min(arr)
        vmax = np.max(arr)
    else:
        vmin = min(np.min(arr), vmin)
        vmax = max(np.max(arr), vmax)
vmin, vmax = torch.torch.full(arr.shape, vmin), torch.torch.full(
    arr.shape, vmax
)

vmin = vmin.detach().cpu().numpy()
np.save("/bigdata/hplsim/aipp/Anna/minmax/vmin_rad.npy", vmin)
vmax = vmax.detach().cpu().numpy()
np.save("/bigdata/hplsim/aipp/Anna/minmax/vmax_rad.npy", vmax)
