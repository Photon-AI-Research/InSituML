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

def plot_3D(particle_tensor, color_component=-1):

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

    plt.show()
    
def plot_per_slice(pc, slice_along, num_slices, comp_of_interest, axs):
    if comp_of_interest != None:
        comp_of_interest = comp_of_interest - 1

    slices = [np.min(pc[:, slice_along]) + (np.max(pc[:, slice_along]) - np.min(pc[:, slice_along])) * i/num_slices for i in range(num_slices)]
    pc_ = np.concatenate((pc, np.zeros((pc.shape[0], 1))), axis=1)

    for ind in range(len(slices)-1):
        pc_[:, -1][(pc_[:, -3]>=slices[ind]) & (pc_[:, -3]<=slices[ind+1])] = ind
    pc_[:, -1][(pc_[:, -3]>=slices[-1])] = len(slices) - 1

    if comp_of_interest != None:
        mean_energy = [np.mean(pc_[:, comp_of_interest][pc_[:,-1]==ind]) for ind in range(len(slices))]
        std_energy = [np.std(pc_[:, comp_of_interest][pc_[:,-1]==ind]) if pc_[:, comp_of_interest][pc_[:,-1]==ind].shape[0] > 1 else None for ind in range(len(slices)) ]
        axs.plot([slice_ for slice_ in slices], mean_energy)
        axs.tick_params(axis='y', which='major', rotation=45)
        axs.grid(True)
        axs.set_xlabel('Z [mm]')
        axs.set_ylabel('Mean Energy [MeV]')
        #axs.legend(prop={'size': 20})


    if comp_of_interest == None:
        num_particles = [pc_[pc_[:,-1]==ind].shape[0] for ind in range(len(slices))]
        axs.plot([slice_ for slice_ in slices], num_particles)
        axs.tick_params(axis='y', which='major', rotation=45)
        axs.grid(True)
        axs.set_xlabel('Z [mm]')
        axs.set_ylabel('Number of particles')
        #axs.legend(prop={'size': 20})
        
def plot_2D(pc, comp_x, comp_y, axs, label_x, label_y):
    axs.scatter(pc[:,comp_x], pc[:,comp_y], s=15, alpha=0.2)
    axs.tick_params(axis='y', which='major', rotation=45)
    axs.grid(True)
    axs.set_xlabel(label_x)
    axs.set_ylabel(label_y)
    #axs.legend(prop={'size': 20})