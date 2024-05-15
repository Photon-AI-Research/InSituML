import torch
from torch.utils.data import Dataset
import numpy as np

from . import data_preprocessing


class PCDataset(Dataset):
    def __init__(
        self,
        items_phase_space,
        items_radiation,
        num_points=-1,
        num_files=1,
        chunk_size=10000,
        species="e_all",
        normalize=False,
        a=0.0,
        b=1.0,
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

        self.get_data_phase_space_by_chunks = (
            data_preprocessing.get_phase_space_by_chunks
        )
        self.get_data_radiation = (
            data_preprocessing.get_radiation_spectra_2_projections
        )
        # self.get_data_radiation = data_preprocessing.get_unit_condition
        self.normalize = normalize
        self.num_points = num_points
        self.a, self.b = a, b
        self.species = species

        # self.vmin_ps, self.vmax_ps = data_preprocessing.get_vmin_vmax_ps(items_phase_space)
        # self.vmin_rad, self.vmax_rad = data_preprocessing.get_vmin_vmax_radiation(items_radiation)

        self.items_file_chunk = []

        if num_files == -1:
            self.items_radiation = items_radiation
            self.items_phase_space = items_phase_space
        else:
            self.items_radiation = items_radiation[:num_files]
            self.items_phase_space = items_phase_space[:num_files]

        self.num_files = len(self.items_radiation)

        self.shapes = []

        for item in items_phase_space:
            shape = data_preprocessing.get_shape(item, species)
            if shape > num_points and num_points != -1:
                self.shapes.append(num_points)
            if shape < num_points or num_points == -1:
                self.shapes.append(shape)

        self.chunk_size = chunk_size

        for ind, shape in enumerate(self.shapes):
            if shape % chunk_size == 0:
                num_chunks_per_file = shape // chunk_size
            else:
                num_chunks_per_file = shape // chunk_size + 1

            for j in range(num_chunks_per_file):
                self.items_file_chunk.append((items_phase_space[ind], j))

        print("\nNumber of simulations: ", self.num_files)
        print("\nNumber of chunks: ", len(self.items_file_chunk))

        print("\nGet min/max from phase space data...")
        for j in range(len(self.items_file_chunk)):
            # print('Chunk ',j)
            arr, _ = self.__getitem__(j)
            # particle_tensor = particle_tensor[~np.isnan(particle_tensor).any(axis=1)]

            arr = arr.detach().cpu().numpy()
            arr = arr[~np.isnan(arr).any(axis=1)]

            if j == 0:
                self.vmin_ps = [np.min(arr[:, i]) for i in range(arr.shape[1])]
                self.vmax_ps = [np.max(arr[:, i]) for i in range(arr.shape[1])]
                # print(self.vmin_ps)
            else:

                self.vmin_ps = [
                    min(np.min(arr[:, i]), self.vmin_ps[i])
                    for i in range(arr.shape[1])
                ]
                self.vmax_ps = [
                    max(np.max(arr[:, i]), self.vmax_ps[i])
                    for i in range(arr.shape[1])
                ]
                # print(self.vmin_ps)

        self.vmin_ps = torch.Tensor(self.vmin_ps).float()
        self.vmax_ps = torch.Tensor(self.vmax_ps).float()

        print("PS Minima: ")
        print("\t", self.vmin_ps)

        print("PS Maxima: ")
        print("\t", self.vmax_ps)

        print("\nGet min/max from radiation data...")
        # self.vmin_rad, self.vmax_rad = data_preprocessing.get_vmin_vmax_radiation(items_radiation, self.chunk_size, self.get_data_radiation)
        # print(self.vmin_rad.shape)
        # self.vmin_rad = self.vmin_rad[:,:2]
        # self.vmax_rad = self.vmax_rad[:,:2]
        path_to_minmax = "/bigdata/hplsim/aipp/Anna/minmax/"
        """

        path_to_minmax = '/bigdata/hplsim/aipp/Anna/minmax/'
        self.vmin_ps, self.vmax_ps, self.a, self.b = torch.from_numpy(np.load(path_to_minmax+'vmin_ps.npy')), torch.from_numpy(np.load(path_to_minmax+'vmax_ps.npy')), torch.Tensor([0.]), torch.Tensor([1.])
        self.vmin_ps = self.vmin_ps
        self.vmax_ps = self.vmax_ps
        """
        vmin_rad, vmax_rad = np.load(path_to_minmax + "vmin_rad.npy"), np.load(
            path_to_minmax + "vmax_rad.npy"
        )
        # print(vmin_rad.shape)
        self.vmin_rad, self.vmax_rad = vmin_rad[0, 0], vmax_rad[0, 0]

        print("PS Minima: ")
        print("\t", self.vmin_ps)

        print("PS Maxima: ")
        print("\t", self.vmax_ps)

        print("Radiation Minima: ")
        print("\t", self.vmin_rad)

        print("Radiation Maxima: ")
        print("\t", self.vmax_rad)

    def __getitem__(self, index):
        ind_rad = self.items_phase_space.index(self.items_file_chunk[index][0])
        return (
            self.get_data_phase_space_by_chunks(
                index,
                items=self.items_file_chunk,
                chunk_size=self.chunk_size,
                species=self.species,
            ),
            self.get_data_radiation(
                ind_rad, items=self.items_radiation, chunk_size=self.chunk_size
            ),
        )

    def __len__(self):
        return self.num_files
