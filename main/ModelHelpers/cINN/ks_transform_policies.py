"""
Definitions of transformation policies for particle or radiation data.
Implement policies as functors!
To be used in the openPMD data producers which load PIConGPU data from disk or via streaming.
"""

from torch import stack as torch_stack
from torch import cat as torch_cat
from torch import float32 as torch_float32
from torch import angle as torch_angle
from torch import abs as torch_abs
from torch import log as torch_log

class BoxesParticlesAttributes:
    """
    """
    pass



class PerpendicularAbsoluteAndPhase:
    """ Transform distributed amplitudes of radiation data to keep only components of the vector perpendicular to the x-direction.
        Also returns the absolute and phase value of the complex radiation amplitude, instead of a complex number.

        Args:
            distributed_amplitudes (torch.Tensor): Radiation data in the shape of (MPI ranks or local chunks, components, frequencies)
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
                res = np.setdiff1d(expanded_slice, np.array([slice_]), assume_unique=True)
                return res                                                                
                
        inv_x = erase_single_index_from_slice(0, distributed_amplitudes.shape[1])                                                                                      
                
        r = distributed_amplitudes[:, inv_x, :]                                           
        
        # Compute the phase (angle) of the complex number                                 
        phase = torch_angle(r)                                                            
        # Compute the absolute value of the complex number
        absolute = torch_abs(r)
        return torch_cat((absolute, phase), dim=1).to(torch_float32) # shape: (MPI ranks or local chunks, 2 * (y,z)-components, frequencies)


class AbsoluteSquare:
    """ Transform distributed amplitudes of radiation data to return the summed absolute square of the distributed amplitude components.

        Args:
            distributed_amplitudes (torch.Tensor): Radiation data in the shape of (MPI ranks or local chunks, components, frequencies)
                That is, radiation data is already along a single direction.

        Returns:
            r (torch.Tensor): sum of absolute square values of components of complex radiation amplitude vectors
    """
    def __call__(self, distributed_amplitudes):
        r = torch_stack([(torch_abs(distributed_amplitudes[:, i_c, :]).to(torch_float32))**2 for i_c in range(distributed_amplitudes.shape[1])], dim=0)
        r = r.sum(dim=0)
        #log transformation
        r = torch_log(r+1)
        return r

