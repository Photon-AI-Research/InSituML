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

path_to_supercells_indices = '/bigdata/hplsim/aipp/Anna/lwfa_supercells/supercells_indices/'
path_to_supercells = '/bigdata/hplsim/aipp/Anna/lwfa_supercell_44_1151_51/'
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

num_supercells = 1
#num_supercells_to_merge = 10

for ind, series in enumerate(all_series[:1]):
    iteration = int(paths_to_simulation_files[ind].split('.')[0].split('_')[-1])
    print('Iteration: ', iteration)
    supercell_files = []
    supercell_filters = []
    tmp_x1_unique = np.load(path_to_supercells_indices+str(iteration)+'_xnum.npy')
    tmp_y1_unique = np.load(path_to_supercells_indices+str(iteration)+'_ynum.npy')
    tmp_z1_unique = np.load(path_to_supercells_indices+str(iteration)+'_znum.npy')
    
    #xnum = (np.random.choice(tmp_x1_unique[num_supercells_to_merge:(-1-num_supercells_to_merge)] // 8, size=num_supercells, replace=False))
    #ynum = (np.random.choice(tmp_y1_unique[num_supercells_to_merge:(-1-num_supercells_to_merge)] // 8, size=num_supercells, replace=False))
    #znum = (np.random.choice(tmp_z1_unique[num_supercells_to_merge:(-1-num_supercells_to_merge)] // 4, size=num_supercells, replace=False))
    
    xnum = 44
    ynum = 1151
    znum = 51

    #for supercell_num_x in range(num_supercells):
    for supercell_num_x in range(num_supercells):
            for supercell_num_y in range(num_supercells):
                for supercell_num_z in range(num_supercells):
                    print('Supercell: '+'; '.join([str(xnum+supercell_num_x), str(ynum+supercell_num_y), str(znum+supercell_num_z)]))
                    supercell_file = open(path_to_supercells
                                            +'_'.join([str(xnum+supercell_num_x), str(ynum+supercell_num_y), str(znum+supercell_num_z)])+'.npy', 'wb')
                    supercell_files.append(supercell_file)

    i = series.iterations[iteration]
    
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

        series.flush() 

        supercell_filters = []
        print('Create filters...')
        for supercell_num_x in range(num_supercells):
            for supercell_num_y in range(num_supercells):
                for supercell_num_z in range(num_supercells):
                    my_filter = (np.logical_and(np.logical_and((x_pos_offset // 8 < xnum+supercell_num_x),
                                                              (y_pos_offset // 8 < ynum+supercell_num_y)),
                                               z_pos_offset // 4 < znum+supercell_num_z))

                    supercell_filters.append(my_filter)

        for supercell_num in range(len(supercell_filters)):
            x_pos_ = x_pos[supercell_filters[supercell_num]]
            y_pos_ = y_pos[supercell_filters[supercell_num]]
            z_pos_ = z_pos[supercell_filters[supercell_num]]

            x_pos_offset_ = x_pos_offset[supercell_filters[supercell_num]]
            y_pos_offset_ = y_pos_offset[supercell_filters[supercell_num]]
            z_pos_offset_ = z_pos_offset[supercell_filters[supercell_num]]

            x_momentum_ = x_momentum[supercell_filters[supercell_num]]
            y_momentum_ = y_momentum[supercell_filters[supercell_num]]
            z_momentum_ = z_momentum[supercell_filters[supercell_num]]


            particle_tensor = np.stack((x_pos_+x_pos_offset_,
                                        y_pos_+y_pos_offset_,
                                        z_pos_+z_pos_offset_,
                                        x_momentum_,
                                        y_momentum_,
                                        z_momentum_), axis=-1)

            #print('Shape: ', particle_tensor.shape)
            if particle_tensor.shape[0] > 0:
                np.save(supercell_files[supercell_num], particle_tensor)
            particle_tensor = None

    for supercell_num in range(len(supercell_files)):
        supercell_files[supercell_num].close()
        

all_files = os.listdir(path_to_supercells)
cells = []
for f in all_files:
    print(path_to_supercells+f)
    cells.append(np.load(path_to_supercells+f))
    
all_cells = np.stack(cells, axis=1)
print('All cells: ', cells.shape)
    
