import matplotlib.pyplot as plt
import seaborn as sns

def plot_reconstructed_data(original, reconstructed, plot_title, cmap = 'gray'):
    columns = len(original)
    fig, ax = plt.subplots(2,columns, figsize=(7*(columns),15))
    fig.suptitle(plot_title, fontsize=60)

    for i in range(columns):
        og = original[i]
        ax[0][i].imshow(og, cmap = cmap)
        ax[0][i].set_title("Original", fontsize = 30)

        recon = reconstructed[i]
        ax[1][i].imshow(recon, cmap = cmap)
        ax[1][i].set_title("Reconstructed", fontsize = 30)
        
    return fig

def plot_reconstructed_data_taskwise(original, task_img_dict, task_ids, plot_title, cmap = 'gray'):
    plt.figure()
    rows = len(task_ids)
    columns = rows + 1
    fig , ax = plt.subplots(rows,columns, figsize=(3*(columns),3*rows))
    fig.suptitle(plot_title, fontsize=60)
    im = None

    #plotting originals
    for r in range(rows):
        if r == 0:
            ax[r,0].set_title("Originals", fontsize = 15)
        ax[r,0].imshow(original[r], cmap = cmap)
        ax[r,0].set_ylabel("Original Task {}".format(task_ids[r] + 1),  fontsize = 15)

    #plotting reconstructed imgs
    for idx, task_id in enumerate(task_ids):
        ax[0,idx+1].set_title("After Task {}".format(task_id + 1), fontsize = 15)
        for r, recon in enumerate(task_img_dict["M 1:{}".format(task_id + 1)]):
            im = ax[r,idx+1].imshow(recon, cmap = cmap)
    
    for r in range(rows):
        for c in range(columns):
            ax[r,c].get_xaxis().set_ticks([])
            ax[r,c].get_yaxis().set_ticks([])
    
    cbar_ax = fig.add_axes([0.04 , 0.15, 0.05, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=30)

    return fig

def plot_heatmap_df(data, plot_title, normalize = True):
    plt.figure()
    if normalize:
        normalized_df= (data-data.min().min())/(data.max().max()-data.min().min())
    else:
        normalized_df = data
    h_map = sns.heatmap(normalized_df, square = True, cmap="RdBu_r", annot=True, fmt ='.2f' , annot_kws={"fontsize":7})
    h_map.set_title(plot_title)
    plt.yticks(rotation=0)

    return h_map