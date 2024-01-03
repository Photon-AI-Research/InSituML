import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import openpmd_api as io
import os
from radiation import RadiationData


filename2= "/lustre/orion/csc380/world-shared/rpausch/002_KHI_withRad_randomInit/simOutput/openPMD/simData_000000.bp"
series2 = io.Series(filename2, io.Access_Type.read_only)

ii = series2.iterations[0]
particles = ii.particles["e_all"]

h1 = particles.particle_patches["extent"]

extent_x = h1["x"].load()
extent_y = h1["y"].load()
extent_z = h1["z"].load()

h4 = particles.particle_patches["offset"]

offset_x = h4["x"].load()
offset_y = h4["y"].load()
offset_z = h4["z"].load()

series2.flush()

# offsets of gpu boxes in pic units
r_offset = np.empty((len(extent_x), 3))
r_offset[:, 0] = offset_x * ii.get_attribute("cell_width")
r_offset[:, 1] = offset_y * ii.get_attribute("cell_height")
r_offset[:, 2] = offset_z * ii.get_attribute("cell_depth")

filename = "/lustre/orion/csc380/world-shared/rpausch/002_KHI_withRad_randomInit/simOutput/radiationOpenPMD/e_radAmplitudes%T.bp"
series = io.Series(filename, io.Access_Type.read_only)

for iteration in range(2,2001):
#for iteration in range(1817, 2001):
    print(iteration)
    i = series.iterations[iteration]

    Amplitude_distributed = i.meshes["Amplitude_distributed"]
    DetectorDirection = i.meshes["DetectorDirection"]
    DetectorFrequency = i.meshes["DetectorFrequency"]

    Dist_Amplitude_x = Amplitude_distributed["x"][:, :, :]
    Dist_Amplitude_y = Amplitude_distributed["y"][:, :, :]
    Dist_Amplitude_z = Amplitude_distributed["z"][:, :, :]

    DetectorDirection_x = DetectorDirection["x"][:, 0, 0]
    DetectorDirection_y = DetectorDirection["y"][:, 0, 0]
    DetectorDirection_z = DetectorDirection["z"][:, 0, 0]

    DetectorFrequency = DetectorFrequency["omega"][0, :, 0]
    series.flush()
    
    n_vec = np.empty((len(DetectorDirection_x), 3))
    n_vec[:, 0] = DetectorDirection_x
    n_vec[:, 1] = DetectorDirection_y
    n_vec[:, 2] = DetectorDirection_z

    # time retardation correction
    phase_offset = np.exp(-1.j * DetectorFrequency[np.newaxis, np.newaxis, :]*  (iteration + np.dot(r_offset, n_vec.T)[:, :, np.newaxis] / 1.0))
    
    Dist_Amplitude_offset_x = (Dist_Amplitude_x / phase_offset)
    Dist_Amplitude_offset_y = (Dist_Amplitude_y / phase_offset)
    Dist_Amplitude_offset_z = (Dist_Amplitude_z / phase_offset)
    
    #index = 0 to get ex vector = [1,0,0]
    index = 0
    amplitude_x = Dist_Amplitude_offset_x[:, index, :]
    amplitude_y = Dist_Amplitude_offset_y[:, index, :]
    amplitude_z = Dist_Amplitude_offset_z[:, index, :]

    # Concatenate along the second axis
    amplitude_concat = np.stack((amplitude_x, amplitude_y, amplitude_z), axis=1)
        
    file_rad_new = '/lustre/orion/csc372/proj-shared/vineethg/khi/part_rad/radiation_002_ex/' + str(iteration)+'.npy'

    if os.path.exists(file_rad_new):
        # File exists, do nothing
        print("File already exists. Skipping dataset creation.")
    else:
        # File does not exist, create the dataset and save it
        np.save(file_rad_new, amplitude_concat)
        print("Dataset saved successfully.")

    if iteration % 100 == 0:
        series.close()
        series = io.Series(filename, io.Access_Type.read_only)