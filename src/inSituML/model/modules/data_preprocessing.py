import numpy as np
import random
import torch

import h5py as h5
import openpmd_api as io

try:
    from . import radiation
except: 
    import radiation


def get_data_phase_space(ind, 
                         items,
                         num_particles=-1,
                         species='e_all'):
    '''
    Load file with 1 particle cloud
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
        num_particles(integer): number of particles to sample from an electron cloud, 
                           if -1 then take electron cloud completely
        species(string): name of particle species to be loaded from the openPMD format
    '''
    
    series = io.Series(items[ind],
                       io.Access.read_only, '{"defer_iteration_parsing": true}')
    #print(items[ind].split('.bp')[-1].split('_')[-1])
    i = series.iterations[int(items[ind].split('.bp')[0].split('_')[-1])].open()
    
    particles = i.particles["b_all"]

    x_pos = particles["position"]["x"]
    y_pos = particles["position"]["y"]
    z_pos = particles["position"]["z"]
    x_pos_offset = particles["positionOffset"]["x"]
    y_pos_offset = particles["positionOffset"]["y"]
    z_pos_offset = particles["positionOffset"]["z"]

    x_momentum = particles["momentum"]["x"]
    y_momentum = particles["momentum"]["y"]
    z_momentum = particles["momentum"]["z"]

    series.flush()

    particle_tensor = np.stack((x_pos+x_pos_offset,
                                y_pos+y_pos_offset,
                                z_pos+z_pos_offset,
                                x_momentum,
                                y_momentum,
                                z_momentum), axis=-1)

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
        items(list of (string, integer)): list of paths to files and number of chunk
                                          to be loaded from a corresponding file
        chunk_size(integer): number of particles to load per time from an electron cloud
                             (a complete cloud is too large)
        species(string): name of particle species to be loaded from the openPMD format
    '''
    
    filename = items[ind][0]
    chunk_num = items[ind][1]
   
    series = io.Series(filename,
                       io.Access.read_only, '{"defer_iteration_parsing": true}')
    i = series.iterations[int(filename.split('.bp')[0].split('_')[-1])].open()
    
    particles = i.particles[species]

    if (get_shape(filename, species) < (chunk_num+1)*chunk_size):
        x_pos = particles["position"]["x"][chunk_num*chunk_size:]
        y_pos = particles["position"]["y"][chunk_num*chunk_size:]
        z_pos = particles["position"]["z"][chunk_num*chunk_size:]
        x_pos_offset = particles["positionOffset"]["x"][chunk_num*chunk_size:]
        y_pos_offset = particles["positionOffset"]["y"][chunk_num*chunk_size:]
        z_pos_offset = particles["positionOffset"]["z"][chunk_num*chunk_size:]

        x_momentum = particles["momentum"]["x"][chunk_num*chunk_size:]
        y_momentum = particles["momentum"]["y"][chunk_num*chunk_size:]
        z_momentum = particles["momentum"]["z"][chunk_num*chunk_size:]

        series.flush()

        particle_tensor = np.stack((x_pos+x_pos_offset,
                                    y_pos+y_pos_offset,
                                    z_pos+z_pos_offset,
                                    x_momentum,
                                    y_momentum,
                                    z_momentum), axis=-1)
    else:
        x_pos = particles["position"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
        y_pos = particles["position"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
        z_pos = particles["position"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
        x_pos_offset = particles["positionOffset"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
        y_pos_offset = particles["positionOffset"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
        z_pos_offset = particles["positionOffset"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]

        x_momentum = particles["momentum"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
        y_momentum = particles["momentum"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
        z_momentum = particles["momentum"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]

        series.flush()

        particle_tensor = np.stack((x_pos+x_pos_offset,
                                    y_pos+y_pos_offset,
                                    z_pos+z_pos_offset,
                                    x_momentum,
                                    y_momentum,
                                    z_momentum), axis=-1)
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
    return torch.flatten(spectra.repeat(chunk_size, 1, 1), start_dim=1)
    
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

def get_unit_condition(ind,
                       items,
                       chunk_size):
    '''
    a simple condition for test
    '''
    c = torch.from_numpy(np.array([1])).float()
    return c.repeat(chunk_size, 1)

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
    #return concatenated_projections.repeat(chunk_size, 1)[:,:2]
    return concatenated_projections.repeat(chunk_size, 1)

def normalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Normalize point from a set of points with vmin(minimum) and vmax(maximum)
    to be in a range [a, b]
    '''
    #print(point.shape, vmin.shape, vmax.shape, a, b)
    return (a + (point - vmin) * (b - a) / ( vmax - vmin))

def denormalize_point(point, vmin, vmax, a=0., b=1.):
    '''
    Denormalize point from range [a, b]
    to be in set of points with vmin(minimum) and vmax(maximum)
    '''
    return ((point - a) * (vmax - vmin) / (b - a) + vmin)

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

def get_vmin_vmax_radiation_np(items,
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
    #return  torch.torch.full(arr.shape, vmin),  torch.torch.full(arr.shape, vmax)
    return vmin, vmax

def get_shape(item, species):
    series = io.Series(item,
                       io.Access.read_only, '{"defer_iteration_parsing": true}')
    i = series.iterations[int(item.split('.bp')[0].split('_')[-1])].open()
    
    particles = i.particles[species]
    charge = particles["charge"][io.Mesh_Record_Component.SCALAR]

    series.flush()
    return charge.get_attribute('shape')

def get_particles_for_plot(filename, num_particles=10000):
    '''
    Extract first num_particles from file for plotting function
    '''

    series = io.Series(filename, io.Access.read_only, '{"defer_iteration_parsing": true}')
    iteration = int(filename.split('.')[0].split('_')[-1])
    i = series.iterations[iteration].open()
    particles = i.particles["e_all"]
    
    x_pos = particles["position"]["x"][:num_particles]
    y_pos = particles["position"]["y"][:num_particles]
    z_pos = particles["position"]["z"][:num_particles]
    x_pos_offset = particles["positionOffset"]["x"][:num_particles]
    y_pos_offset = particles["positionOffset"]["y"][:num_particles]
    z_pos_offset = particles["positionOffset"]["z"][:num_particles]

    x_momentum = particles["momentum"]["x"][:num_particles]
    y_momentum = particles["momentum"]["y"][:num_particles]
    z_momentum = particles["momentum"]["z"][:num_particles]

    series.flush()
    
    particle_tensor = np.stack((x_pos+x_pos_offset,
                                y_pos+y_pos_offset,
                                z_pos+z_pos_offset,
                                x_momentum,
                                y_momentum,
                                z_momentum), axis=-1)

    return particle_tensor

def time_to_hotvec(time, timesteps):
    hotvec = np.zeros((len(timesteps)))
    hotvec[timesteps.index(time)] += 1
    return hotvec

def get_timestep(path_to_phase_space):
    return path_to_phase_space.split('/')[-1].split('.')[0].split('_')[0]

#new
def get_time(path_to_phase_space):
    return path_to_phase_space.split('/')[-1].split('.')[0].split('_')[0]
