import torch
from utilities import ( Normalizer, 
                       random_sample,
                       filter_dims,
                       inspect_and_select,
                       create_force_density_plots,
                       create_momentum_density_plots)
import wandb
import argparse
from args import get_args
from args_transform import main_args_transform
import numpy as np
import ast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@inspect_and_select
def load_particles(file_pattern="/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/particle_002/{}.npy",
                box_indices =[3,19],
                t0 = 1900,
                t1 = 2001,
                t_step = 2,
                particles_to_sample=150000,
                normalizer = Normalizer(),
                norm_method = 'mean_6d',
                property_ = "momentum_force"):
    particles = []
    for time_step in range(t0, t1, t_step):
        
        #things a man has to do..
        p = np.load(file_pattern.format(time_step), allow_pickle = True)
        p = [p[idx] for idx in box_indices]
        p = [normalizer.normalize_data(element, method=norm_method) for element in p]
        p = np.array(p, dtype=object)
        p = [random_sample(element, sample_size=particles_to_sample) for element in p]
        p = torch.from_numpy(np.array(p, dtype = np.float32))
        
        #filter dims
        p = filter_dims[property_](p)
        
        particles.append([time_step, p])
    
    return particles

@inspect_and_select
def plot_particles(model, particles, wandb_obj, box_indices=[3,19]):
    
    for [ts, particles] in particles:
        for idx, phase_space in enumerate(particles):
            _, predictions = model(torch.unsqueeze(phase_space, 0).to(device))
            
            momentum_pred = predictions[0][:,:3]
            momentum_gt = phase_space[:,:3]

            force_pred = predictions[0][:,3:6]
            force_gt = phase_space[:,3:6]
            
            timeInfo = f"timestep_{ts}_box_{box_indices[idx]}"
            
            create_momentum_density_plots(*(momentum_gt.T.tolist()+ momentum_pred.T.tolist()),
                                          path="",
                                          t=timeInfo, 
                                          wandb = wandb_obj)

            create_force_density_plots(*(force_gt.T.tolist() + force_pred.T.tolist()),
                                          path="",
                                          t=timeInfo, 
                                          wandb = wandb_obj)
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="""For running plotting predictions from a loaded model"""
    )


    parser.add_argument('--box_indices',
                        type=str,
                        default='[19,6]',
                        help="Box indices to plot")

    parser.add_argument('--t_step',
                        type=int,
                        default=50,
                        help="no help")

    parser.add_argument('--model_path',
                        type=str,
                        default=".",
                        help="no help")
    
    args = vars(get_args(parser))
    args["box_indices"] = ast.literal_eval(args["box_indices"])    
    args = main_args_transform(args)
    
    # load the trained model
    model = args["model"]

    model.load_state_dict(torch.load(args["model_path"]))
    model.to(device)
    model.eval()
    config = {k:v for k, v in args.items() if k in ["box_indices", "model_path",
                                                        "t_step", "t0", "t1"]}

    wandb.init(config = config, 
               project = f'newruns_{args["project_kw"]}',
               name="_".join([str(k)+"_"+str(v) for k,v in config.items()]))
    
    args["wandb_obj"] = wandb
    particles = load_particles(**args)
    plot_particles(particles=particles,**args)
