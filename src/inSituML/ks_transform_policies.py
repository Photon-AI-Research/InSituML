"""
Definitions of transformation policies for particle or radiation data.
Implement policies as functors!
To be used in the openPMD data producers which load PIConGPU data
from disk or via streaming.
"""

import numpy as np
from torch import stack as torch_stack
from torch import transpose as torch_transpose
from torch import sum as torch_sum
from torch import cat as torch_cat
from torch import float32 as torch_float32
from torch import angle as torch_angle
from torch import abs as torch_abs
from torch import log as torch_log


#################################
# Particle transform policies #
#################################
class BoxesAttributesParticles:
    """
    Transform particle data to shape
    (GPUs, phaseSpaceComponents, particlesPerGPU)

    Args:
        loaded_particles (torch.tensor): particle data in the shape of
        (phaseSpaceComponents, GPUs, particlesPerGPU)

    Returns:
        (torch.tensor): can be a map or a copy of the transformed data.
        pytorch decides.
    """

    def __call__(self, loaded_particles):
        return torch_transpose(loaded_particles, 0, 1)


class ParticlesAttributes:
    """Transform particle data to shape (particles, phaseSpaceComponents).
    That is, it performs a sum over all local subvolumes (=GPUs).

    Args:
        loaded_particles (torch.tensor): particle data in the shape of
         (phaseSpaceComponents, GPUs, particlesPerGPU)

    Returns:
        (torch.tensor):
    """

    def __call__(self, loaded_particles):
        summedParticles = torch_sum(loaded_particles, dim=1)
        return torch_transpose(summedParticles, 0, 1)


##################################
# Radiation transform policies #
##################################
class PerpendicularAbsoluteAndPhase:
    """
    Transform distributed amplitudes of radiation data to keep only
     components of the vector perpendicular to the x-direction.
    Also returns the absolute and phase value of the complex radiation
     amplitude, instead of a complex number.

    Args:
        distributed_amplitudes (torch.Tensor):
         Radiation data in the shape of (MPI ranks or local chunks,
          components, frequencies)
            That is, radiation data is already along a single direction.

    Returns:
        r (torch.Tensor):
    """

    def __call__(self, distributed_amplitudes):
        def erase_single_index_from_slice(slice_, len):
            if isinstance(slice_, slice):
                return slice(None)
            else:
                expanded_slice = np.array(range(len))
                # remove the single index from the indexes
                res = np.setdiff1d(
                    expanded_slice, np.array([slice_]), assume_unique=True
                )
                return res

        inv_x = erase_single_index_from_slice(
            0, distributed_amplitudes.shape[1]
        )

        r = distributed_amplitudes[:, inv_x, :]

        # Compute the phase (angle) of the complex number
        phase = torch_angle(r)
        # Compute the absolute value of the complex number
        absolute = torch_abs(r)
        return torch_cat((absolute, phase), dim=1).to(torch_float32)
        # shape: (MPI ranks or local chunks, 2 * (y,z)-components, frequencies)


class AbsoluteSquare:
    """
    Transform distributed amplitudes of radiation data to return
     the summed absolute square of the distributed amplitude components.

    Args:
        distributed_amplitudes (torch.Tensor):
         Radiation data in the shape of
          (MPI ranks or local chunks, components, frequencies)
            That is, radiation data is already along a single direction.

    Returns:
        r (torch.Tensor): sum of absolute square values of
         components of complex radiation amplitude vectors,
          shape (MPI ranks or local chunks, frequencies)
    """

    def __call__(self, distributed_amplitudes):
        r = torch_stack(
            [
                (
                    torch_abs(distributed_amplitudes[:, i_c, :]).to(
                        torch_float32
                    )
                )
                ** 2
                for i_c in range(distributed_amplitudes.shape[1])
            ],
            dim=0,
        )
        r = torch_sum(r, dim=0)
        # log transformation
        r = torch_log(r + 1.0e-9)
        return r


class AbsoluteSquareSumRanks:
    """
    Same as AbsoluteSquare but return sum over all MPI ranks.

    Args:
        distributed_amplitudes (torch.Tensor): Radiation data in the shape of
         (MPI ranks or local chunks, components, frequencies)
            That is, radiation data is already along a single direction.

    Returns:
        r (torch.Tensor): Absolute square values of components of complex
         radiation amplitude vectors summed over all components
          and all MPI ranks.
    """

    def __call__(self, distributed_amplitudes):
        absSquare = AbsoluteSquare()
        r = absSquare(distributed_amplitudes)
        return torch_sum(r, dim=0)
