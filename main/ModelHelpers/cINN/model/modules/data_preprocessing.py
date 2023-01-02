import numpy as np
import random
import torch

import h5py as h5
import openpmd_api as io

import radiation

def get_data_phase_space(ind, 
                         items,
                         num_particles=-1):
    '''
    Load file with 1 particle cloud
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
        num_particles(integer): number of particles to sample from an electron cloud, 
                           if -1 then take electron cloud completely
    '''
    
    series = io.Series(items[ind],
                       io.Access.read_only)
    #print(items[ind].split('.bp')[-1].split('_')[-1])
    i = series.iterations[int(items[ind].split('.bp')[0].split('_')[-1])]
    
    particles = i.particles["b_all"]

    particle_tensor = np.stack((particles["position"]["x"],
                                particles["position"]["y"],
                                particles["position"]["z"],
                                particles["momentum"]["x"],
                                particles["momentum"]["y"],
                                particles["momentum"]["z"]), axis=-1)

    if num_particles == -1:
        return particle_tensor
    else:
        inds = random.sample(list(range(0, particle_tensor.shape[0])), num_particles)
        return torch.from_numpy(particle_tensor[inds, :]).float()
    
def get_phase_space_by_chunks(ind, 
                                   items,
                                   chunk_size=100,
                                   species='e_all'):
    '''
    Load file with 1 particle cloud
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
        num_particles(integer): number of particles to sample from an electron cloud, 
                           if -1 then take electron cloud completely
    '''
    
    filename = items[ind][0]
    chunk_num = items[ind][1]
   
    series = io.Series(filename,
                       io.Access.read_only)
    i = series.iterations[int(filename.split('.bp')[0].split('_')[-1])]
    
    particles = i.particles[species]

    if (get_shape(filename, species) < (chunk_num+1)*chunk_size):
        particle_tensor = np.stack((particles["position"]["x"][chunk_num*chunk_size:],
                                    particles["position"]["y"][chunk_num*chunk_size:],
                                    particles["position"]["z"][chunk_num*chunk_size:],
                                    particles["momentum"]["x"][chunk_num*chunk_size:],
                                    particles["momentum"]["y"][chunk_num*chunk_size:],
                                    particles["momentum"]["z"][chunk_num*chunk_size:],
                                    particles["momentumPrev1"]["x"][chunk_num*chunk_size:],
                                    particles["momentumPrev1"]["y"][chunk_num*chunk_size:],
                                    particles["momentumPrev1"]["z"][chunk_num*chunk_size:]), axis=-1)
    else:
        particle_tensor = np.stack((particles["position"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["position"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["position"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentum"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentum"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentum"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentumPrev1"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentumPrev1"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentumPrev1"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]), axis=-1)

    #particle_tensor = particle_tensor[~np.isnan(particle_tensor).any(axis=1)]
    return torch.from_numpy(particle_tensor).float()
    

def h5_tree(val, pre=''):
    items = len(val)
    for key, val in val.items():
        items -= 1
        if items == 0:
            # the last item
            if type(val) == h5._hl.group.Group:
                print(pre + '└── ' + key)
                h5_tree(val, pre+'    ')
            else:
                print(pre + '└── ' + key + ' (%d)' % len(val))
        else:
            if type(val) == h5._hl.group.Group:
                print(pre + '├── ' + key)
                h5_tree(val, pre+'│   ')
            else:
                print(pre + '├── ' + key + ' (%d)' % len(val))
    
def get_testdata_radiation(ind, 
                             items,
                             chunk_size):
    '''
    Load complex amplitudes of 1 radiation file from HZDR cloud
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
        chunk_size(int): number of particles in 1 batch, is needed to dublicate amplitudes for each particle
    '''

    f = h5.File(items[ind], 'r')
    
    x = torch.from_numpy(np.stack((f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('x_Im')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('x_Re')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('y_Im')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('y_Re')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('z_Im')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('z_Re')[:]), axis=2)).float().squeeze()
    return x.repeat(chunk_size, 1, 1, 1)

def get_radiation_spectra(ind, 
                          items,
                          chunk_size):
    '''
    Load spectra of 1 radiation file
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
        chunk_size(int): number of particles in 1 batch, is needed to dublicate amplitudes for each particle
    '''

    spectra = torch.from_numpy((radiation.RadiationData(items[ind])).get_Spectra()).float()
    return spectra.repeat(chunk_size, 1, 1)
    
def get_radiation_spectra_intergrated_over_directions(ind, 
                                                     items,
                                                     chunk_size):
    '''
    Load intergrated over directions spectra of 1 radiation file
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
        chunk_size(int): number of particles in 1 batch, is needed to dublicate amplitudes for each particle
    '''

    spectra = (radiation.RadiationData(items[ind])).get_Spectra()
    integrated_spectra = torch.from_numpy(np.sum(spectra, axis=0)).float()
    return integrated_spectra.repeat(chunk_size, 1)

def get_radiation_spectra_intergrated_over_frequencies(ind, 
                                                       items,
                                                       chunk_size):
    '''
    Load intergrated over frequencies spectra of 1 radiation file
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
        chunk_size(int): number of particles in 1 batch, is needed to dublicate amplitudes for each particle
    '''

    spectra = (radiation.RadiationData(items[ind])).get_Spectra()
    integrated_spectra = torch.from_numpy(np.sum(spectra, axis=1)).float()
    return integrated_spectra.repeat(chunk_size, 1)

def get_radiation_spectra_2_projections(ind,
                                        items,
                                        chunk_size):
    '''
    Load radiation the spectra, integrated over directions and the spectra, integrated over frequencies
        and concatenate them into 1 tensor.
    Important: use only if confident, that there is the same mesh in radiation simulations
    '''

    spectra = (radiation.RadiationData(items[ind])).get_Spectra()
    concatenated_projections = torch.from_numpy(np.concatenate((np.sum(spectra, axis=0), np.sum(spectra, axis=1)))).float()
    return concatenated_projections.repeat(chunk_size, 1)

def normalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Normalize point from a set of points with vmin(minimum) and vmax(maximum)
    to be in a range [a, b]
    '''
    return (a + (point - vmin) * (b - a) / ( vmax - vmin))

def denormalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Denormalize point from range [a, b]
    to be in set of points with vmin(minimum) and vmax(maximum)
    '''
    return ((point - a) * (vmax - vmin) / (b - a) + vmin)

def get_vmin_vmax_ps(arr):
    '''
    Find minima/maxima in all columns among a complete data(all simulation files)
    Args:
        items(list of string): list of paths to all electron clouds
    
    returns torch tensors with minima and maxima
    '''
    
    for ind,item in enumerate(items):
        arr = get_data_phase_space(ind, 
                       items,
                       num_particles=-1)

        if item == items[0]:
            vmin = [np.min(arr[:, i]) for i in range(arr.shape[1])]
            vmax = [np.max(arr[:, i]) for i in range(arr.shape[1])]
        else:
            vmin = [min(np.min(arr[:, i]), vmin[i]) for i in range(arr.shape[1])]
            vmax = [max(np.max(arr[:, i]), vmax[i]) for i in range(arr.shape[1])]
    return torch.Tensor(vmin), torch.Tensor(vmax)

def get_vmin_vmax_radiation(items,
                            chunk_size,
                            get_radiation_data):
    '''
    Find minima/maxima in all radiation simulations for normalization
    Args:
        items(list of string): list of paths to all electron clouds
        chunk_size(int): number of particles in 1 batch, is needed to dublicate amplitudes for each particle
        get_radiation_data(function): function to load and preprocess radiation data
    
    returns torch tensors with minima and maxima
    '''
    for ind,item in enumerate(items):
        arr = get_radiation_data(ind, 
                                 items,
                                 chunk_size)
        arr = arr.detach().cpu().numpy()
    
        if item == items[0]:
            vmin = np.min(arr)
            vmax = np.max(arr)
        else:
            vmin = min(np.min(arr), vmin)
            vmax = max(np.max(arr), vmax)
    return  torch.torch.full(arr.shape, vmin),  torch.torch.full(arr.shape, vmax)

def get_shape(item, species):
    series = io.Series(item,
                       io.Access.read_only)
    i = series.iterations[int(item.split('.bp')[0].split('_')[-1])]
    
    particles = i.particles[species]
    charge = particles["charge"][io.Mesh_Record_Component.SCALAR]

    series.flush()
    return charge.get_attribute('shape')