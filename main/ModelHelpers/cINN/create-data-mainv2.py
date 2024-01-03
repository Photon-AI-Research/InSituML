import os
import openpmd_api as io
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('model')

from modules.visualizations import plot_3D, plot_2D, plot_per_slice


print("openPMD-api: {}"
      .format(io.__version__))
print("openPMD-api backend variants: {}"
      .format(io.variants))

#path_to_all_simulations = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/001_KHI_withRad/simOutput/openPMD/"
path_to_all_simulations = "/lustre/orion/csc380/world-shared/rpausch/002_KHI_withRad_randomInit/simOutput/openPMD/"

paths_to_simulation_files = [path_to_all_simulations + directory for directory in os.listdir(path_to_all_simulations)]

paths_to_simulation_files = sorted(paths_to_simulation_files, key=lambda x: os.path.basename(x))

#paths_to_simulation_files = paths_to_simulation_files[772:]

all_series = []
for f in paths_to_simulation_files:
    series = io.Series(f,
                           io.Access.read_only)
    print("Read a Series with openPMD standard version %s" %
          series.openPMD)

    print("The Series contains {0} iterations:".format(len(series.iterations)))
    for i in series.iterations:
        print("\t {0}".format(i))
    print("")
    
    all_series.append(series)


for ind, series in enumerate(all_series):
    print(ind)
    
    iteration = int(paths_to_simulation_files[ind].split('.')[0].split('_')[-1])
    i = series.iterations[iteration]
    
    # print(f"Iteration {iteration} contains {len(i.meshes)} meshes:")
    # for m in i.meshes:
    #     print(f"\t {m}")
    # print("")
    # print(f"Iteration {iteration} contains {len(i.meshes)} particle species:")
    # for ps in i.particles:
    #     print(f"\t {ps}")
    #     print("With records:")
    #     for r in i.particles[ps]:
    #         print(f"\t {r}")
           
    particles = i.particles["e_all"]
    #num_particles = 10000
    
    h1 = particles.particle_patches["extent"]
    extent_x = h1["x"].load()
    extent_y = h1["y"].load()
    extent_z = h1["z"].load()

    h2 = particles.particle_patches["numParticles"][io.Mesh_Record_Component.SCALAR]
    numParticles = h2.load()

    h3 = particles.particle_patches["numParticlesOffset"][io.Mesh_Record_Component.SCALAR]
    numParticlesOffset = h3.load()

    h4 = particles.particle_patches["offset"]

    offset_x = h4["x"].load()
    offset_y = h4["y"].load()
    offset_z = h4["z"].load()
    
    series.flush()
    particles_new = []
    for id_gpu in range (numParticlesOffset.shape[0]):
        print("GPU id:", id_gpu)
    
        x_pos = particles["position"]["x"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles]
        y_pos = particles["position"]["y"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles]
        z_pos = particles["position"]["z"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles]
        x_pos_offset = particles["positionOffset"]["x"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles]
        y_pos_offset = particles["positionOffset"]["y"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles]
        z_pos_offset = particles["positionOffset"]["z"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles]

        x_momentum = particles["momentum"]["x"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles],
        y_momentum = particles["momentum"]["y"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles],
        z_momentum = particles["momentum"]["z"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]  #[:num_particles],

        x_momentumPrev1 = particles["momentumPrev1"]["x"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]
        y_momentumPrev1 = particles["momentumPrev1"]["y"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]
        z_momentumPrev1 = particles["momentumPrev1"]["z"][numParticlesOffset[id_gpu] : numParticlesOffset[id_gpu] + numParticles[id_gpu]]

        series.flush()

        x_force = x_momentum - x_momentumPrev1
        y_force = y_momentum - y_momentumPrev1
        z_force = z_momentum - z_momentumPrev1

        particle_tensor = np.stack((x_pos+x_pos_offset,
                                    y_pos+y_pos_offset,
                                    z_pos+z_pos_offset,
                                    x_momentum,
                                    y_momentum,
                                    z_momentum,
                                    x_force,
                                    y_force,
                                    z_force), axis=-1)

    
        particle_tensor = particle_tensor[~np.isnan(particle_tensor).any(axis=1)]
        print(f"Number of particles with non NaN values: {particle_tensor.shape[0]}")

        # print('particle_tensor',particle_tensor.shape)

        particles_new.append(particle_tensor)

    
    
    file_path = '/lustre/orion/csc372/proj-shared/vineethg/khi/part_rad/particle_002/' +str(iteration)+'.npy'

    print(file_path)

    if os.path.exists(file_path):
        # File exists, do nothing
        print("File already exists. Skipping dataset creation.")
    else:
        # File does not exist, create the dataset and save it
        np.save(file_path, particles_new)
        print("Dataset saved successfully.")


    series.close()