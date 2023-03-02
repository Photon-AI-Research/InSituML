# Copyright 2016-2022 Richard Pausch
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np


class RadiationDataStream:
    def __init__(self, stream_reader):
        """
        Open references to hdf5 file to access radiation data.
        This constructor opens h5py references to the radiation data
        and thus allows easy access to the complex amplitudes from the
        Lienard-Wiechert potential.
        """
        self._iteration = False
        self._stream_reader = stream_reader
        # extract time step
        self.timestep = 0

    def cache_next(self):
        self._iteration = self._stream_reader.get_next_data()
        self.timestep = self._iteration['iteration_index']
        return self._iteration is not None

    def empty_cache(self):
        self._iteration = False

    @property
    def cache_is_empty(self):
        return self._iteration is False

    @property
    def iteration(self):
        if self.cache_is_empty:
            raise ValueError(
                'please call `cache_next` before reading an iteration.')
        return self._iteration

    def close(self):
        self._stream_reader.close()

    def __del__(self):
        try:
            self.close()
        except AttributeError:
            pass

    @property
    def Ax_Re(self):
        return self.iteration['meshes']['Amplitude']['x_Re'].data

    @property
    def Ax_Im(self):
        return self.iteration['meshes']['Amplitude']['x_Im'].data

    @property
    def Ay_Re(self):
        return self.iteration['meshes']['Amplitude']['y_Re'].data

    @property
    def Ay_Im(self):
        return self.iteration['meshes']['Amplitude']['y_Im'].data

    @property
    def Az_Re(self):
        return self.iteration['meshes']['Amplitude']['z_Re'].data

    @property
    def Az_Im(self):
        return self.iteration['meshes']['Amplitude']['z_Im'].data

    @property
    def convert_to_SI(self):
        """Conversion factor for spectra from PIC units to SI units."""
        return self.iteration['meshes']['Amplitude']['x_Re'].unit_si

    def get_Amplitude_x(self):
        """Returns the complex amplitudes in x-axis."""
        return ((self.Ax_Re[...] + 1j * self.Ax_Im[...])[:, :, 0] *
                np.sqrt(self.convert_to_SI))

    def get_Amplitude_y(self):
        """Returns the complex amplitudes in y-axis."""
        return ((self.Ay_Re[...] + 1j * self.Ay_Im[...])[:, :, 0] *
                np.sqrt(self.convert_to_SI))

    def get_Amplitude_z(self):
        """Returns the complex amplitudes in z-axis."""
        return ((self.Az_Re[...] + 1j * self.Az_Im[...])[:, :, 0] *
                np.sqrt(self.convert_to_SI))

    def get_Spectra(self):
        """Returns real spectra in [Js]."""
        return (np.abs(self.get_Amplitude_x())**2 +
                np.abs(self.get_Amplitude_y())**2 +
                np.abs(self.get_Amplitude_z())**2)

    def get_Polarization_X(self):
        """Returns real spectra for x-polarization in [Js]."""
        return np.abs(self.get_Amplitude_x())**2

    def get_Polarization_Y(self):
        """Returns real spectra for y-polarization in [Js]."""
        return np.abs(self.get_Amplitude_y())**2

    def get_Polarization_Z(self):
        """Returns real spectra for z-polarization in [Js]."""
        return np.abs(self.get_Amplitude_z())**2

    def get_omega(self):
        """Returns frequency 'omega' of spectrum in [s^-1]."""
        omega_h = self.iteration['meshes']['DetectorFrequency']['omega']
        return omega_h.data[0, :, 0] * omega_h.unit_si

    def get_vector_n(self):
        """Returns the unit vector 'n' of the observation directions."""
        n_h = self.iteration['meshes']['DetectorDirection']
        n_x = n_h['x'].data[:, 0, 0] * n_h['x'].unit_si
        n_y = n_h['y'].data[:, 0, 0] * n_h['y'].unit_si
        n_z = n_h['z'].data[:, 0, 0] * n_h['z'].unit_si
        n_vec = np.empty((len(n_x), 3))
        n_vec[:, 0] = n_x
        n_vec[:, 1] = n_y
        n_vec[:, 2] = n_z
        return n_vec
