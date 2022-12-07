import openpmd_api as io
import numpy as np
import random
import torch

import h5py as h5

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
    
def get_data_phase_space_by_chunks(ind, 
                                   items,
                                   chunk_size=100):
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
    
    particles = i.particles["b_all"]
    
    
    
    if (get_shape(filename) < (chunk_num+1)*chunk_size):
        particle_tensor = np.stack((particles["position"]["x"][chunk_num*chunk_size:],
                                    particles["position"]["y"][chunk_num*chunk_size:],
                                    particles["position"]["z"][chunk_num*chunk_size:],
                                    particles["momentum"]["x"][chunk_num*chunk_size:],
                                    particles["momentum"]["y"][chunk_num*chunk_size:],
                                    particles["momentum"]["z"][chunk_num*chunk_size:]), axis=-1)
    else:
        particle_tensor = np.stack((particles["position"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["position"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["position"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentum"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentum"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size],
                                    particles["momentum"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]), axis=-1)
    '''
    f = open(str(chunk_num*chunk_size)+'_'+str((chunk_num+1)*chunk_size)+".txt", "a")

    for i in range(particle_tensor.shape[0]):
        f.write(" ".join([str(particle_tensor[i,k]) for k in range(6)]))
    f.close()
    '''
    #print(particle_tensor.dtype)
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
    
def get_data_radiation(ind, 
                       items,
                      chunk_size):
    '''
    Load file with 1 radiation file
    Args:
        ind(integer): number of path in list of paths to files
        items(list of string): list of paths to files
    '''

    f = h5.File(items[ind], 'r')
    '''
    dataAmpl_x_Im = f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('x_Im')[:]
    dataAmpl_x_Re = f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('x_Re')[:]
    dataAmpl_y_Im = f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('y_Im')[:]
    dataAmpl_y_Re = f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('y_Re')[:]
    dataAmpl_z_Im = f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('z_Im')[:]
    dataAmpl_z_Re = f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('z_Re')[:]

    dataDetector_x = f.get('data').get('68500').get('DetectorMesh').get('DetectorDirection').get('x')[:]
    dataDetector_y = f.get('data').get('68500').get('DetectorMesh').get('DetectorDirection').get('y')[:]
    dataDetector_z = f.get('data').get('68500').get('DetectorMesh').get('DetectorDirection').get('z')[:]

    dataFreq = f.get('data').get('68500').get('DetectorMesh').get('DetectorFrequency').get('omega')[:]
    '''
    
    x = torch.from_numpy(np.stack((f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('x_Im')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('x_Re')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('y_Im')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('y_Re')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('z_Im')[:],
                    f.get('data').get('68500').get('DetectorMesh').get('Amplitude').get('z_Re')[:]), axis=2)).float().squeeze()
    return x.repeat(chunk_size,1,1,1)
    
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

def get_vmin_vmax_radiation(items, chunk_size):
    '''
    Find minima/maxima in all radiation simulations for normalization
    Args:
        items(list of string): list of paths to all electron clouds
    
    returns torch tensors with minima and maxima
    '''
    for ind,item in enumerate(items):
        arr = get_data_radiation(ind, 
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

def get_shape(item):
    series = io.Series(item,
                       io.Access.read_only)
    i = series.iterations[int(item.split('.bp')[0].split('_')[-1])]
    
    particles = i.particles["b_all"]
    charge = particles["charge"][io.Mesh_Record_Component.SCALAR]

    series.flush()
    return charge.get_attribute('shape')