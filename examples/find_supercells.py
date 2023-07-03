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

num_supercells = 3
    
for ind, series in enumerate(all_series):
    iteration = int(paths_to_simulation_files[ind].split('.')[0].split('_')[-1])
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

    chunk_size = 100000000
    filesize = data_preprocessing.get_shape(paths_to_simulation_files[ind], "e_all")
    print('Total number of particles: ', filesize)
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
        else:
            print('Reached end of file...')
            exit_loop = True
            x_pos = particles["position"]["x"][chunk_num*chunk_size:]
            y_pos = particles["position"]["y"][chunk_num*chunk_size:]
            z_pos = particles["position"]["z"][chunk_num*chunk_size:]
            x_pos_offset = particles["positionOffset"]["x"][chunk_num*chunk_size:]
            y_pos_offset = particles["positionOffset"]["y"][chunk_num*chunk_size:]
            z_pos_offset = particles["positionOffset"]["z"][chunk_num*chunk_size:]

        series.flush() 
        if chunk_num==0:
            tmp_x1_unique = np.unique(x_pos_offset)
            tmp_y1_unique = np.unique(y_pos_offset)
            tmp_z1_unique = np.unique(z_pos_offset)
        else:
            tmp_x1_unique = np.concatenate((tmp_x1_unique, np.unique(x_pos_offset)))
            tmp_y1_unique = np.concatenate((tmp_y1_unique, np.unique(y_pos_offset)))
            tmp_z1_unique = np.concatenate((tmp_z1_unique, np.unique(z_pos_offset)))
            tmp_x1_unique = np.unique(tmp_x1_unique)
            tmp_y1_unique = np.unique(tmp_y1_unique)
            tmp_z1_unique = np.unique(tmp_z1_unique)

        print("Macrocells in x: ", tmp_x1_unique)
        print("Macrocells in y: ", tmp_y1_unique)
        print("Macrocells in z: ", tmp_z1_unique)

    np.save(path_to_supercells_indices+str(iteration)+'_xnum.npy', tmp_x1_unique)
    np.save(path_to_supercells_indices+str(iteration)+'_ynum.npy', tmp_y1_unique)
    np.save(path_to_supercells_indices+str(iteration)+'_znum.npy', tmp_z1_unique)