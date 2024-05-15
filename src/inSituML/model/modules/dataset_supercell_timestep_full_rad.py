import torch
from torch.utils.data import Dataset
import numpy as np

from . import data_preprocessing


class PCDataset(Dataset):
    def __init__(
        self, items_phase_space, items_radiation, normalize=False, a=0.0, b=1.0
    ):
        """
        Prepare dataset
        Args:
            items_phase_space(list of string): list of paths to files with particle phase space data
            items_radiation(list of string): list of paths to files with radiation, the order of paths should
                                             correspond the order of paths in items_phase_space
            num_points(integer): number of points to sample from each electron cloud,
                                 if -1 then take a complete electron cloud
            num_files(integer): number of files to take for a dataset
            chunk_size(integer): number of particles to load per time
                                 (a complete point cloud does not pass into the memory)
            species(string): name of particle species to be loaded from openPMD file
            normalize(boolean): True if normalize each point to be in range [a, b]
        """

        self.normalize = normalize
        self.a, self.b = a, b

        self.items_ps = items_phase_space
        # len_dataset = sum([np.loadtxt(item).shape[0] for item in ite])
        self.items_rad = items_radiation
        self.get_data_radiation = data_preprocessing.get_radiation_spectra

        if normalize:
            for j, item_ps in enumerate(self.items_ps):
                arr = np.loadtxt(item_ps)
                if j == 0:
                    self.vmin_ps = [
                        np.min(arr[:, i]) for i in range(arr.shape[1])
                    ]
                    self.vmax_ps = [
                        np.max(arr[:, i]) for i in range(arr.shape[1])
                    ]
                else:

                    self.vmin_ps = [
                        min(np.min(arr[:, i]), self.vmin_ps[i])
                        for i in range(arr.shape[1])
                    ]
                    self.vmax_ps = [
                        max(np.max(arr[:, i]), self.vmax_ps[i])
                        for i in range(arr.shape[1])
                    ]

            self.vmin_ps = torch.Tensor(self.vmin_ps).float()
            self.vmax_ps = torch.Tensor(self.vmax_ps).float()

            print("PS Minima: ")
            print("\t", self.vmin_ps)

            print("PS Maxima: ")
            print("\t", self.vmax_ps)

            self.vmin_rad_, self.vmax_rad_ = (
                data_preprocessing.get_vmin_vmax_radiation_np(
                    self.items_rad, 1, self.get_data_radiation
                )
            )

            print("Radiation Minima: ")
            print("\t", self.vmin_rad_)

            print("Radiation Maxima: ")
            print("\t", self.vmax_rad_)

    def __getitem__(self, index):
        x = np.loadtxt(self.items_ps[index])
        self.vmin_rad, self.vmax_rad = torch.full(
            (x.shape[0], 65536), self.vmin_rad_
        ), torch.full((x.shape[0], 65536), self.vmax_rad_)
        return (
            torch.Tensor(x).float(),
            self.get_data_radiation(index, self.items_rad, x.shape[0]),
        )

    def __len__(self):
        return len(self.items_ps)
