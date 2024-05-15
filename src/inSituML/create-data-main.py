import os
import openpmd_api as io
import numpy as np
import matplotlib.pyplot as plt

import sys

sys.path.append("model")

from modules.visualizations import plot_3D, plot_2D, plot_per_slice


print("openPMD-api: {}".format(io.__version__))
print("openPMD-api backend variants: {}".format(io.variants))

path_to_all_simulations = "/bigdata/hplsim/production/KHI_for_GB_MR/runs/001_KHI_withRad/simOutput/openPMD/"
paths_to_simulation_files = [
    path_to_all_simulations + directory
    for directory in os.listdir(path_to_all_simulations)
]

# for f in paths_to_simulation_files:
# print(f)

paths_to_simulation_files = sorted(
    paths_to_simulation_files, key=lambda x: os.path.basename(x)
)
# print(paths_to_simulation_files)

all_series = []
for f in paths_to_simulation_files:
    series = io.Series(f, io.Access.read_only)
    print("Read a Series with openPMD standard version %s" % series.openPMD)

    print("The Series contains {0} iterations:".format(len(series.iterations)))
    for i in series.iterations:
        print("\t {0}".format(i))
    print("")

    all_series.append(series)


for ind, series in enumerate(all_series):
    print(ind)

    iteration = int(
        paths_to_simulation_files[ind].split(".")[0].split("_")[-1]
    )
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
    # num_particles = 10000

    x_pos = particles["position"]["x"][:]  # [:num_particles]
    y_pos = particles["position"]["y"][:]  # [:num_particles]
    z_pos = particles["position"]["z"][:]  # [:num_particles]
    x_pos_offset = particles["positionOffset"]["x"][:]  # [:num_particles]
    y_pos_offset = particles["positionOffset"]["y"][:]  # [:num_particles]
    z_pos_offset = particles["positionOffset"]["z"][:]  # [:num_particles]

    x_momentum = particles["momentum"]["x"][:]  # [:num_particles],
    y_momentum = particles["momentum"]["y"][:]  # [:num_particles],
    z_momentum = particles["momentum"]["z"][:]  # [:num_particles],

    x_momentumPrev1 = particles["momentumPrev1"]["x"][:]
    y_momentumPrev1 = particles["momentumPrev1"]["y"][:]
    z_momentumPrev1 = particles["momentumPrev1"]["z"][:]

    # added
    series.flush()

    x_force = x_momentum - x_momentumPrev1
    y_force = y_momentum - y_momentumPrev1
    z_force = z_momentum - z_momentumPrev1

    # print('x_force', x_force.max())
    # print('x_force', x_force.min())

    # series.flush()

    particle_tensor = np.stack(
        (
            x_pos + x_pos_offset,
            y_pos + y_pos_offset,
            z_pos + z_pos_offset,
            x_momentum,
            y_momentum,
            z_momentum,
            x_force,
            y_force,
            z_force,
        ),
        axis=-1,
    )

    # particle_tensor = np.stack((x_pos+x_pos_offset,
    #                             y_pos+y_pos_offset,
    #                             z_pos+z_pos_offset,
    #                             x_momentum[0],
    #                             y_momentum[0],
    #                             z_momentum[0]), axis=-1)

    particle_tensor = particle_tensor[~np.isnan(particle_tensor).any(axis=1)]
    print(
        f"Number of particles with non NaN values: {particle_tensor.shape[0]}"
    )

    print("particle_tensor", particle_tensor.shape)

    file_path = (
        "/bigdata/hplsim/aipp/Jeyhun/khi/particle/" + str(iteration) + ".npy"
    )

    if os.path.exists(file_path):
        # File exists, do nothing
        print("File already exists. Skipping dataset creation.")
    else:
        # File does not exist, create the dataset and save it
        np.save(file_path, particle_tensor)
        print("Dataset saved successfully.")

print("FINISHED")
