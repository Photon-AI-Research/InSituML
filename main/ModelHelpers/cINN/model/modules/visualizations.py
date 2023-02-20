import numpy as np
import torch
import wandb

import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as tick
from matplotlib import cm

plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20
plt.rcParams['lines.linewidth']=6
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

import data_preprocessing

def plot_3D(particle_tensor, color_component=-1, return_fig=True):

    fig = plt.figure(figsize=(24,10))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    comp = color_component

    norm = matplotlib.colors.Normalize(vmin=np.min(particle_tensor[:, comp]),
                                   vmax=np.max(particle_tensor[:, comp]))

    ax.scatter(particle_tensor[:, 0], particle_tensor[:, 1], particle_tensor[:, 2], 
           c=plt.cm.jet(norm(particle_tensor[:, comp])), s=20, alpha=0.5)
    ax.view_init(40, 200)

    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    cb = plt.colorbar(m, shrink=0.6, pad=0.01)
    cb.ax.tick_params(labelsize=23)
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel('Momentum Z', fontsize=25)

    ax.set_xlabel('x', fontsize=25, linespacing=5.2)
    #ax.set_xlabel('', fontsize=25, linespacing=5.2)
    ax.set_ylabel('y', fontsize=25, linespacing=5.2)
    ax.set_zlabel('z', fontsize=25, linespacing=5.2)
    ax.dist = 10

    ax.xaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_tick_params(labelsize=23)
    ax.zaxis.set_tick_params(labelsize=23)
    ax.xaxis.labelpad=30
    ax.yaxis.labelpad=30
    ax.zaxis.labelpad=30

    ax.tick_params(axis='both', which='major', pad=10)

    ax.xaxis._axinfo['label']['space_factor'] = 5.0
    ax.yaxis._axinfo['label']['space_factor'] = 5.0
    ax.zaxis._axinfo['label']['space_factor'] = 5.0

    ax.set_title('GT', fontsize=25)
    if return_fig:
        return fig
    else:
        plt.show()

def plot_per_slice(pc, slice_along, num_slices, comp_of_interest, axs, label='Number of particles'):
    if comp_of_interest is not None:
        comp_of_interest = comp_of_interest - 1

    slices = [np.min(pc[:, slice_along]) + (np.max(pc[:, slice_along]) - np.min(pc[:, slice_along])) * i/num_slices for i in range(num_slices)]
    pc_ = np.concatenate((pc, np.zeros((pc.shape[0], 1))), axis=1)

    for ind in range(len(slices)-1):
        pc_[:, -1][(pc_[:, slice_along]>=slices[ind]) & (pc_[:, slice_along]<=slices[ind+1])] = ind
    pc_[:, -1][(pc_[:, slice_along]>=slices[-1])] = len(slices) - 1

    if comp_of_interest is not None:
        mean_energy = [np.mean(pc_[:, comp_of_interest][pc_[:,-1]==ind]) if (pc_[:, comp_of_interest][pc_[:,-1]==ind]).shape[0] > 1 else None for ind in range(len(slices))]
        #std_energy = [np.std(pc_[:, comp_of_interest][pc_[:,-1]==ind]) if pc_[:, comp_of_interest][pc_[:,-1]==ind].shape[0] > 1 else None for ind in range(len(slices)) ]
        axs.plot([slice_ for slice_ in slices], mean_energy)
        axs.tick_params(axis='y', which='major', rotation=45)
        axs.grid(True)
        axs.set_xlabel('Z')
        axs.set_ylabel(label)
        #axs.legend(prop={'size': 20})

    if comp_of_interest == None:
        num_particles = [pc_[pc_[:,-1]==ind].shape[0] for ind in range(len(slices))]
        axs.plot([slice_ for slice_ in slices], num_particles)
        axs.tick_params(axis='y', which='major', rotation=45)
        axs.grid(True)
        axs.set_xlabel('Z')
        axs.set_ylabel(label)
        #axs.legend(prop={'size': 20})

def plot_per_slice_GTandRec(pc, pc_pred, slice_along, num_slices, comp_of_interest, axs, label='Number of particles'):
    if pc_pred is not None:
        slices_pred = [np.min(pc_pred[:, slice_along]) + (np.max(pc_pred[:, slice_along]) - np.min(pc_pred[:, slice_along])) * i/num_slices for i in range(num_slices)]
        pc_pred_ = np.concatenate((pc_pred, np.zeros((pc_pred.shape[0], 1))), axis=1)

        for ind in range(len(slices_pred)-1):
            pc_pred_[:, -1][(pc_pred_[:, -3]>=slices_pred[ind]) & (pc_pred_[:, -3]<=slices_pred[ind+1])] = ind
        pc_pred_[:, -1][(pc_pred_[:, -3]>=slices_pred[-1])] = len(slices_pred) - 1
    
    if comp_of_interest is not None:
        comp_of_interest = comp_of_interest - 1

    slices = [np.min(pc[:, slice_along]) + (np.max(pc[:, slice_along]) - np.min(pc[:, slice_along])) * i/num_slices for i in range(num_slices)]
    pc_ = np.concatenate((pc, np.zeros((pc.shape[0], 1))), axis=1)

    for ind in range(len(slices)-1):
        pc_[:, -1][(pc_[:, slice_along]>=slices[ind]) & (pc_[:, slice_along]<=slices[ind+1])] = ind
    pc_[:, -1][(pc_[:, slice_along]>=slices[-1])] = len(slices) - 1

    if comp_of_interest is not None:
        mean_energy = [np.mean(pc_[:, comp_of_interest][pc_[:,-1]==ind]) if (pc_[:, comp_of_interest][pc_[:,-1]==ind]).shape[0] > 1 else None for ind in range(len(slices))]
        std_energy = [np.std(pc_[:, comp_of_interest][pc_[:,-1]==ind]) if pc_[:, comp_of_interest][pc_[:,-1]==ind].shape[0] > 1 else None for ind in range(len(slices)) ]

        axs.plot([slice_ for slice_ in slices], mean_energy, label="Groundtruth")
        if pc_pred is not None:
            mean_energy_pred = [np.mean(pc_pred_[:, comp_of_interest+1][pc_pred_[:,-1]==ind]) if (pc_pred_[:, comp_of_interest+1][pc_pred_[:,-1]==ind]).shape[0] > 1 else None for ind in range(len(slices_pred))]
            std_energy_pred = [np.std(pc_pred_[:, comp_of_interest+1][pc_pred_[:,-1]==ind]) if pc_pred_[:, comp_of_interest+1][pc_pred_[:,-1]==ind].shape[0] > 1 else None for ind in range(len(slices_pred)) ]
            axs.plot([slice_ for slice_ in slices_pred], mean_energy_pred, label="Reconstruction")
            
        axs.tick_params(axis='y', which='major', rotation=45)
        axs.grid(True)
        axs.set_xlabel('Z')
        axs.set_ylabel(label)
        axs.legend(prop={'size': 20})


    if comp_of_interest == None:
        num_particles = [pc_[pc_[:,-1]==ind].shape[0] for ind in range(len(slices))]
        axs.plot([slice_ for slice_ in slices], num_particles)
        if pc_pred is not None:
            num_particles_pred = [pc_pred_[pc_pred_[:,-1]==ind].shape[0] for ind in range(len(slices_pred))]
            axs.plot([slice_ for slice_ in slices_pred], num_particles_pred)
        axs.tick_params(axis='y', which='major', rotation=45)
        axs.grid(True)
        axs.set_xlabel('Z')
        axs.set_ylabel(label)
        #axs.legend(prop={'size': 20})
        
def plot_2D(pc, comp_x, comp_y, axs, label_x, label_y):
    axs.scatter(pc[:,comp_x], pc[:,comp_y], s=15, alpha=0.2)
    axs.tick_params(axis='y', which='major', rotation=45)
    axs.grid(True)
    axs.set_xlabel(label_x)
    axs.set_ylabel(label_y)
    #axs.legend(prop={'size': 20})
#fix labels
def plot_2D_GTandRec(pc, pc_pred, comp_x, comp_y, axs, label_x, label_y):
    axs.scatter(pc[:,comp_x], pc[:,comp_y], s=15, alpha=0.2)
    if pc_pred is not None:
        axs.scatter(pc_pred[:,comp_x], pc_pred[:,comp_y], s=15, alpha=0.2)
    axs.tick_params(axis='y', which='major', rotation=45)
    axs.grid(True)
    axs.set_xlabel(label_x)
    axs.set_ylabel(label_y)

def plot_3D_GTandRec(pc_groundtruth, pc_reconstructed, color_component=-1):
    fig = plt.figure(figsize=(24,10))
    ax = fig.add_subplot(1, 2, 1, projection='3d')

    comp = color_component

    norm = matplotlib.colors.Normalize(vmin=np.min(pc_groundtruth[:, comp]),
                                   vmax=np.max(pc_groundtruth[:, comp]))

    ax.scatter(pc_groundtruth[:, 0], pc_groundtruth[:, 1], pc_groundtruth[:, 2], 
           c=plt.cm.jet(norm(pc_groundtruth[:, comp])), s=20, alpha=0.5)
    ax.view_init(40, 200)

    m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    cb = plt.colorbar(m, shrink=0.6, pad=0.01)
    cb.ax.tick_params(labelsize=23)
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel('Momentum Z', fontsize=25)

    ax.set_xlabel('x', fontsize=25, linespacing=5.2)
    #ax.set_xlabel('', fontsize=25, linespacing=5.2)
    ax.set_ylabel('y', fontsize=25, linespacing=5.2)
    ax.set_zlabel('z', fontsize=25, linespacing=5.2)
    ax.dist = 10

    ax.xaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_tick_params(labelsize=23)
    ax.zaxis.set_tick_params(labelsize=23)
    ax.xaxis.labelpad=30
    ax.yaxis.labelpad=30
    ax.zaxis.labelpad=30

    ax.tick_params(axis='both', which='major', pad=10)

    ax.xaxis._axinfo['label']['space_factor'] = 5.0
    ax.yaxis._axinfo['label']['space_factor'] = 5.0
    ax.zaxis._axinfo['label']['space_factor'] = 5.0

    ax.set_title('GT', fontsize=25)

    ax = fig.add_subplot(1, 2, 2, projection='3d')

    comp = color_component
    if pc_reconstructed is not None:

        norm = matplotlib.colors.Normalize(vmin=np.min(pc_reconstructed[:, comp]),
                                    vmax=np.max(pc_reconstructed[:, comp]))

        ax.scatter(pc_reconstructed[:, 0], pc_reconstructed[:, 1], pc_reconstructed[:, 2], 
            c=plt.cm.jet(norm(pc_reconstructed[:, comp])), s=20, alpha=0.5)
        ax.view_init(40, 200)

        m = cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
        m.set_array([])
        cb = plt.colorbar(m, shrink=0.6, pad=0.01)
    cb.ax.tick_params(labelsize=23)
    cb.ax.get_yaxis().labelpad = 15
    cb.ax.set_ylabel('Momentum Z', fontsize=25)

    ax.set_xlabel('x', fontsize=25, linespacing=5.2)
    #ax.set_xlabel('', fontsize=25, linespacing=5.2)
    ax.set_ylabel('y', fontsize=25, linespacing=5.2)
    ax.set_zlabel('z', fontsize=25, linespacing=5.2)
    ax.dist = 10

    ax.xaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_tick_params(labelsize=23)
    ax.zaxis.set_tick_params(labelsize=23)
    ax.xaxis.labelpad=30
    ax.yaxis.labelpad=30
    ax.zaxis.labelpad=30

    ax.tick_params(axis='both', which='major', pad=10)

    ax.xaxis._axinfo['label']['space_factor'] = 5.0
    ax.yaxis._axinfo['label']['space_factor'] = 5.0
    ax.zaxis._axinfo['label']['space_factor'] = 5.0

    ax.set_title('Reconstruction', fontsize=25)

    #plt.show()
    return fig

def log_plots(test_pointclouds, test_radiations, model):
    for ind, test_pointcloud in enumerate(test_pointclouds):
        log_each_plot(test_pointcloud, test_radiations[ind], model)

def log_min_max_each(test_pointclouds, test_radiations, model):
    for ind, test_pointcloud in enumerate(test_pointclouds):
        log_min_max(test_pointcloud, test_radiations[ind], model)

def getOverlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))

def log_min_max(test_pointcloud, test_radiation, model):
    num_particles = 1000000
    path_to_minmax = '/bigdata/hplsim/aipp/Anna/minmax/'
    num_supercells_to_merge = 10
    supercell = [int(k) for k in test_pointcloud.split('/')[-1].split('.')[0].split('_')[1:]]
    iteration = test_pointcloud.split('/')[-1].split('.')[0].split('_')[0]

    radiation_tensor = data_preprocessing.get_radiation_spectra_2_projections(0, [test_radiation], num_particles)
    vmin_rad, vmax_rad = np.load(path_to_minmax+'vmin_rad.npy'),np.load(path_to_minmax+'vmax_rad.npy')

    radiation_tensor = data_preprocessing.normalize_point(radiation_tensor, 
                                                      torch.full(radiation_tensor.shape, vmin_rad[0,0]), 
                                                      torch.full(radiation_tensor.shape, vmax_rad[0,0]), 
                                                      torch.full(radiation_tensor.shape, 0.), 
                                                      torch.full(radiation_tensor.shape, 1.))

    gt = np.load(test_pointcloud)

    pred_pointcloud_full = model.sample_pointcloud(radiation_tensor.to('cuda'), num_particles)
    pred_pointcloud_full = pred_pointcloud_full.detach().cpu().numpy()
    
    gt_filter = np.logical_and((np.logical_and(np.logical_and((gt[:,0] // 8 < supercell[0]+num_supercells_to_merge),
                                                        (gt[:,1] // 8 < supercell[1]+num_supercells_to_merge)),
                                        gt[:,2] // 4 < supercell[2]+num_supercells_to_merge)),
                
                
                (np.logical_and(np.logical_and((gt[:,0] // 8 > supercell[0]-num_supercells_to_merge),
                                                        (gt[:,1] // 8 > supercell[1]-num_supercells_to_merge)),
                                        gt[:,2] // 4 > supercell[2]-num_supercells_to_merge)))

    mins_gt = [np.min(gt[gt_filter][:,i]) for i in range(gt.shape[1])]
    maxs_gt = [np.max(gt[gt_filter][:,i]) for i in range(gt.shape[1])]

    mins = [np.min(pred_pointcloud_full[:,i]) for i in range(pred_pointcloud_full.shape[1])]
    maxs = [np.max(pred_pointcloud_full[:,i]) for i in range(pred_pointcloud_full.shape[1])]
    #print(supercell)
    labels = ['x', 'xp', 'y', 'yp', 'z', 'zp']
    overlaps = []
    for ind,elem in enumerate(mins):
        overlaps.append(getOverlap([mins_gt[ind], maxs_gt[ind]], [mins[ind], mins[ind]]) > 0)
    wandb.log({'Supercell '+' '.join([str(k) for k in supercell])+', Overlap with prediction ': int(all(overlaps) == True)})

def print_min_max(test_pointcloud, test_radiation, model):
    num_particles = 1000000
    path_to_minmax = '/bigdata/hplsim/aipp/Anna/minmax/'
    num_supercells_to_merge = 10
    supercell = [int(k) for k in test_pointcloud.split('/')[-1].split('.')[0].split('_')[1:]]
    iteration = test_pointcloud.split('/')[-1].split('.')[0].split('_')[0]

    radiation_tensor = data_preprocessing.get_radiation_spectra_2_projections(0, [test_radiation], num_particles)
    vmin_rad, vmax_rad = np.load(path_to_minmax+'vmin_rad.npy'),np.load(path_to_minmax+'vmax_rad.npy')

    radiation_tensor = data_preprocessing.normalize_point(radiation_tensor, 
                                                      torch.full(radiation_tensor.shape, vmin_rad[0,0]), 
                                                      torch.full(radiation_tensor.shape, vmax_rad[0,0]), 
                                                      torch.full(radiation_tensor.shape, 0.), 
                                                      torch.full(radiation_tensor.shape, 1.))

    gt = np.load(test_pointcloud)

    pred_pointcloud_full = model.sample_pointcloud(radiation_tensor.to('cuda'), num_particles)
    pred_pointcloud_full = pred_pointcloud_full.detach().cpu().numpy()
    
    gt_filter = np.logical_and((np.logical_and(np.logical_and((gt[:,0] // 8 < supercell[0]+num_supercells_to_merge),
                                                        (gt[:,1] // 8 < supercell[1]+num_supercells_to_merge)),
                                        gt[:,2] // 4 < supercell[2]+num_supercells_to_merge)),
                
                
                (np.logical_and(np.logical_and((gt[:,0] // 8 > supercell[0]-num_supercells_to_merge),
                                                        (gt[:,1] // 8 > supercell[1]-num_supercells_to_merge)),
                                        gt[:,2] // 4 > supercell[2]-num_supercells_to_merge)))

    mins_gt = [np.min(gt[gt_filter][:,i]) for i in range(gt.shape[1])]
    maxs_gt = [np.max(gt[gt_filter][:,i]) for i in range(gt.shape[1])]

    mins = [np.min(pred_pointcloud_full[:,i]) for i in range(pred_pointcloud_full.shape[1])]
    maxs = [np.max(pred_pointcloud_full[:,i]) for i in range(pred_pointcloud_full.shape[1])]
    #print(supercell)
    labels = ['x', 'xp', 'y', 'yp', 'z', 'zp']
    for ind,elem in enumerate(mins):
        print(labels[ind],'\t\tGT', '\t\tPred')
        print('min:\t', mins_gt[ind], '\t',mins[ind])
        print('max:\t', maxs_gt[ind], '\t',maxs[ind])

def log_each_plot(test_pointcloud, test_radiation, model):
    supercell = [int(k) for k in test_pointcloud.split('/')[-1].split('.')[0].split('_')[1:]]
    iteration = test_pointcloud.split('/')[-1].split('.')[0].split('_')[0]
    #pointcloud_tensor = data_preprocessing.get_particles_for_plot(test_pointcloud, num_particles)
    pointcloud_tensor = np.load(test_pointcloud)
    
    num_particles = 1000000
    idx = np.random.randint(pointcloud_tensor.shape[0], size=num_particles)
    pointcloud_tensor = pointcloud_tensor[idx, :]
    radiation_tensor = data_preprocessing.get_radiation_spectra_2_projections(0, [test_radiation], num_particles)

    labels_x = ['x', 'y', 'z']
    labels_y = ['xp', 'yp', 'zp']

    pred_pointcloud_full = model.sample_pointcloud(radiation_tensor.to(model.device), radiation_tensor.shape[0])
    pred_pointcloud_full = pred_pointcloud_full.detach().cpu().numpy()

    fig = plot_3D(pred_pointcloud_full[np.random.choice(pred_pointcloud_full.shape[0], 2000, replace=False)])
    image = wandb.Image(fig)
    wandb.log({"PC Complete": image})
    plt.close()

    #print(pred_pointcloud_full[(pred_pointcloud_full[:,0] // 8 == supercell[0])].shape)

    #filter = np.logical_and(np.logical_and((pred_pointcloud_full[:,0] == pred_pointcloud_full[0,0]),
    #                                                  (pred_pointcloud_full[:,1] == pred_pointcloud_full[0,1])),
    #                                   pred_pointcloud_full[:,2] == pred_pointcloud_full[0,2])
    
    my_filter = np.logical_and((np.logical_and(np.logical_and((x_pos_offset // 8 < xnum[supercell_num]+num_supercells_to_merge),
                                                      (y_pos_offset // 8 < ynum[supercell_num]+num_supercells_to_merge)),
                                       z_pos_offset // 4 < znum[supercell_num]+num_supercells_to_merge)),
            
            
            (np.logical_and(np.logical_and((x_pos_offset // 8 > xnum[supercell_num]-num_supercells_to_merge),
                                                      (y_pos_offset // 8 > ynum[supercell_num]-num_supercells_to_merge)),
                                       z_pos_offset // 4 > znum[supercell_num]-num_supercells_to_merge)))
    if (pred_pointcloud_full[my_filter].shape[0] > 0):
        pred_pointcloud = pred_pointcloud_full[my_filter]
    else:
        pred_pointcloud = None
    
    fig = plot_3D_GTandRec(pointcloud_tensor, pred_pointcloud)
    image = wandb.Image(fig)
    wandb.log({"PC, Iteration "+iteration+", supercell: "+' '.join([str(k) for k in supercell]): image})
    plt.close()

    slice_along = 2 #z: 2
    num_slices = 100
    comp_of_interest = -3
    figsize1 = 60
    figsize2 = 12
    
    fig, axs = plt.subplots(1, 4, figsize=(figsize1,figsize2))
    plot_per_slice_GTandRec(pointcloud_tensor, pred_pointcloud,
                            slice_along, num_slices, comp_of_interest,
                            axs[0], label='Number of particles')

    for i in range(1,4,1):
        plot_per_slice_GTandRec(pointcloud_tensor, pred_pointcloud,
                            slice_along, num_slices, comp_of_interest=-1*i,
                            axs=axs[i], label='Mean ' + labels_y[-1*i])
    image = wandb.Image(fig)
    wandb.log({"Slicewise, Iteration "+iteration+", supercell: "+' '.join([str(k) for k in supercell]): image})
    plt.close()
    
    figsize1 = 50
    figsize2 = 12
    fig, axs = plt.subplots(1, 3, figsize=(figsize1,figsize2))

    for i in range(3):
        plot_2D_GTandRec(pointcloud_tensor, pred_pointcloud,
                         comp_x=i, comp_y=i+3, axs=axs[i],
                         label_x=labels_x[i], label_y=labels_y[i])

    image = wandb.Image(fig)
    wandb.log({"2D Projections, Iteration "+iteration+", supercell: "+' '.join([str(k) for k in supercell]): image})
    plt.close()