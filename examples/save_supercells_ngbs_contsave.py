import openpmd_api as io
import numpy as np
import matplotlib.pyplot as plt

print("openPMD-api: {}"
      .format(io.__version__))
print("openPMD-api backend variants: {}"
      .format(io.variants))

import os

import sys
sys.path.append('../main/ModelHelpers/cINN/model/modules')
import data_preprocessing

path_to_supercells_indices = '/bigdata/hplsim/aipp/Anna/supercells_indices/'
path_to_supercells = '/bigdata/hplsim/aipp/Anna/lwfa_2cells_unitSI/'
path_to_all_simulations = "/bigdata/hplsim/production/LWFA_radiation_new/LWFArad_data_example/LWFArad_data_example/openPMD/"
paths_to_simulation_files = [path_to_all_simulations + directory for directory in os.listdir(path_to_all_simulations)]

for f in paths_to_simulation_files:
    print(f)
    
all_series = []
for f in paths_to_simulation_files:
    series = io.Series(f, io.Access.read_only)
    print("Read a Series with openPMD standard version %s" %
          series.openPMD)

    print("The Series contains {0} iterations:".format(len(series.iterations)))
    for i in series.iterations:
        print("\t {0}".format(i))
    print("")
    
    all_series.append(series)

#num_supercells = 1
num_neighbours = 1
xnum, ynum, znum = 44, 1151, 51
    
for ind, series in enumerate(all_series):
    iteration = int(paths_to_simulation_files[ind].split('.')[0].split('_')[-1])

    supercell_files = []
    supercell_filters = []
    supercell_tensors = []
    tmp_x1_unique = np.load(path_to_supercells_indices+str(iteration)+'_xnum.npy')
    tmp_y1_unique = np.load(path_to_supercells_indices+str(iteration)+'_ynum.npy')
    tmp_z1_unique = np.load(path_to_supercells_indices+str(iteration)+'_znum.npy')
    
    #xnum = (np.random.choice(tmp_x1_unique[num_supercells_to_merge:(-1-num_supercells_to_merge)] // 8, size=num_supercells, replace=False))
    #ynum = (np.random.choice(tmp_y1_unique[num_supercells_to_merge:(-1-num_supercells_to_merge)] // 8, size=num_supercells, replace=False))
    #znum = (np.random.choice(tmp_z1_unique[num_supercells_to_merge:(-1-num_supercells_to_merge)] // 4, size=num_supercells, replace=False))
    '''
    if num_neighbours > 0:
        for supercell_num_x in range(-1*num_neighbours,num_neighbours):
            for supercell_num_y in range(-1*num_neighbours,num_neighbours):
                for supercell_num_z in range(-1*num_neighbours,num_neighbours):
                    print('Supercell: '+'; '.join([str(xnum + supercell_num_x), str(ynum+supercell_num_y), str(znum+supercell_num_z)]))
                    supercell_file = open(path_to_supercells
                                            +'_'.join([str(iteration), str(xnum + supercell_num_x), str(ynum + supercell_num_y), str(znum + supercell_num_z)])+'.npy', 'a')
                    supercell_files.append(supercell_file)

                    #np.save(path_to_supercells+'_'.join([str(iteration), str(xnum + supercell_num_x), str(ynum + supercell_num_y), str(znum + supercell_num_z)])+'.npy',
                    #        np.zeros((1,6)))
    '''
    if num_neighbours == 0:
        print('Supercell: '+'; '.join([str(xnum), str(ynum), str(znum)]))
        supercell_file = open(path_to_supercells
                                +'_'.join([str(iteration), str(xnum), str(ynum), str(znum)])+'.npy', 'wb')
        supercell_files.append(supercell_file)

    i = series.iterations[iteration]
    '''
    print(f"Iteration {iteration} contains {len(i.meshes)} meshes:")
    for m in i.meshes:
        print(f"\t {m}")
    print("")
    print(f"Iteration {iteration} contains {len(i.meshes)} particle species:")
    for ps in i.particles:
        print(f"\t {ps}")
        print("With records:")
        for r in i.particles[ps]:
            print(f"\t {r}")
    '''      
    particles = i.particles["e_all"]
    filesize = data_preprocessing.get_shape(paths_to_simulation_files[ind], "e_all")
    chunk_size = 100000000
    #filesize = 10000
    #chunk_size = 450
    exit_loop = False
    for chunk_num in range(100000000):
        if exit_loop:
            break
        print('Process chunk ', chunk_num)
        print(chunk_num*chunk_size, ' particles are processed')
        if (filesize >= (chunk_num+1)*chunk_size):
            x_pos = particles["position"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            y_pos = particles["position"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            z_pos = particles["position"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            x_pos_offset = particles["positionOffset"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            y_pos_offset = particles["positionOffset"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            z_pos_offset = particles["positionOffset"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]

            x_momentum = particles["momentum"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            y_momentum = particles["momentum"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            z_momentum = particles["momentum"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            
            x_momentum_prev = particles["momentumPrev1"]["x"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            y_momentum_prev = particles["momentumPrev1"]["y"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            z_momentum_prev = particles["momentumPrev1"]["z"][chunk_num*chunk_size:(chunk_num+1)*chunk_size]
            
            weighting = particles["weighting"][io.Mesh_Record_Component.SCALAR]
            dataWeightings = weighting.load_chunk()[chunk_num*chunk_size:(chunk_num+1)*chunk_size]

        else:
            print('Reached end of file...')
            exit_loop = True
            x_pos = particles["position"]["x"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            y_pos = particles["position"]["y"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            z_pos = particles["position"]["z"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            x_pos_offset = particles["positionOffset"]["x"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            y_pos_offset = particles["positionOffset"]["y"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            z_pos_offset = particles["positionOffset"]["z"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]

            x_momentum = particles["momentum"]["x"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            y_momentum = particles["momentum"]["y"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            z_momentum = particles["momentum"]["z"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]            
            
            x_momentum_prev = particles["momentumPrev1"]["x"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            y_momentum_prev = particles["momentumPrev1"]["y"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            z_momentum_prev = particles["momentumPrev1"]["z"][chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]
            
            weighting = particles["weighting"][io.Mesh_Record_Component.SCALAR]
            dataWeightings = weighting.load_chunk()[chunk_num*chunk_size:chunk_num*chunk_size  + (filesize - chunk_num*chunk_size)]

        series.flush()

        supercell_filters = []
        if num_neighbours == 0:
            my_filter = np.logical_and(np.logical_and((x_pos_offset // 8 == xnum),
                                                      (y_pos_offset // 8 == ynum)),
                                       z_pos_offset // 4 == znum)
            supercell_filters.append(my_filter)
        if num_neighbours > 0:
            for supercell_num_x in range(-1*num_neighbours,num_neighbours):
                for supercell_num_y in range(-1*num_neighbours,num_neighbours):
                    for supercell_num_z in range(-1*num_neighbours,num_neighbours):
                        my_filter = np.logical_and(np.logical_and((x_pos_offset // 8 == xnum+supercell_num_x),
                                                                  (y_pos_offset // 8 == ynum+supercell_num_y)),
                                                   z_pos_offset // 4 == znum+supercell_num_z)
                        
                        x_pos_ = x_pos[my_filter]
                        y_pos_ = y_pos[my_filter]
                        z_pos_ = z_pos[my_filter]

                        x_pos_offset_ = x_pos_offset[my_filter]
                        y_pos_offset_ = y_pos_offset[my_filter]
                        z_pos_offset_ = z_pos_offset[my_filter]

                        x_momentum_ = x_momentum[my_filter]
                        y_momentum_ = y_momentum[my_filter]
                        z_momentum_ = z_momentum[my_filter]
                        
                        x_momentum_prev_ = x_momentum_prev[my_filter]
                        y_momentum_prev_ = y_momentum_prev[my_filter]
                        z_momentum_prev_ = z_momentum_prev[my_filter]
                        
                        dataWeightings_ = dataWeightings[my_filter]
        
                        x_pos_ *= particles["position"]["x"].unit_SI
                        y_pos_ *= particles["position"]["y"].unit_SI
                        z_pos_ *= particles["position"]["z"].unit_SI

                        x_momentum_ *= particles["momentum"]["x"].unit_SI
                        y_momentum_ *= particles["momentum"]["y"].unit_SI
                        z_momentum_ *= particles["momentum"]["z"].unit_SI
                        x_momentum_prev_ *= particles["momentumPrev1"]["x"].unit_SI
                        y_momentum_prev_ *= particles["momentumPrev1"]["y"].unit_SI
                        z_momentum_prev_ *= particles["momentumPrev1"]["z"].unit_SI

                        '''
                        particle_tensor = np.stack((x_pos_+x_pos_offset_,
                                                    y_pos_+y_pos_offset_,
                                                    z_pos_+z_pos_offset_,
                                                    x_momentum_,
                                                    y_momentum_,
                                                    z_momentum_,
                                                    x_momentum_prev_,
                                                    y_momentum_prev_,
                                                    z_momentum_prev_), axis=-1)
                                                    #dataWeightings_), axis=-1)
                        '''
                        
                        particle_tensor = np.stack((x_pos_+x_pos_offset_,
                                                    y_pos_+y_pos_offset_,
                                                    z_pos_+z_pos_offset_,
                                                    x_momentum_,
                                                    y_momentum_,
                                                    z_momentum_,
                                                    x_momentum_-x_momentum_prev_,
                                                    y_momentum_-y_momentum_prev_,
                                                    z_momentum_-z_momentum_prev_,
                                                    dataWeightings_,
                                                    x_momentum_prev_,
                                                    y_momentum_prev_,
                                                    z_momentum_prev_), axis=-1)
                        
                        curr_file = path_to_supercells+'_'.join([str(iteration), str(xnum + supercell_num_x), str(ynum + supercell_num_y), str(znum + supercell_num_z)])+'.npy'
                        file_descr = open(curr_file, 'a')
                        np.savetxt(file_descr, particle_tensor)
                        file_descr.close()
'''
if num_neighbours > 0:
    all_files = os.listdir(path_to_supercells)
    cells = []
    for f in all_files:
        print(path_to_supercells+f)
        cells.append(np.load(path_to_supercells+f))

    all_cells = np.concatenate([arr for arr in cells], axis=0)
    all_file = open(path_to_supercells+'all_cells.npy', 'wb')
    np.save(all_cells, path_to_supercells+'all_cells.npy')
    all_file.close()
'''