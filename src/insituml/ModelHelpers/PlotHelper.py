import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch

def prepare_data_for_plot(task_id, original, predicted , dim , image_index, min_norm, max_norm):
    original_np = original.detach().cpu().numpy()[0]
    predicted_np = predicted.detach().cpu().numpy()[0]
    
    #original_np = original_np * (max_norm - min_norm) + min_norm
    #predicted_np = predicted_np * (max_norm - min_norm) + min_norm
    
    dim_map = {
        "x":0,
        "y":1,
        "z":2
    }
    
    index_to_use = 0#dim_map[dim]
    fig, (ax1,ax2) = plt.subplots(1,2)
    fig.suptitle("Prediction Batch {}".format(task_id))
    ax1.imshow(original_np[index_to_use][image_index], cmap ="jet")
    ax1.set_title("Original")
    ax2.imshow(predicted_np[index_to_use][image_index], cmap ="jet")
    ax2.set_title("Predicted")
    wandb.log({"Image {} Jet Scale".format(task_id):fig})
    
    ax1.imshow(original_np[index_to_use][image_index], cmap ="gray")
    ax2.imshow(predicted_np[index_to_use][image_index], cmap ="gray")
    wandb.log({"Image {} Gray Scale".format(task_id):fig})