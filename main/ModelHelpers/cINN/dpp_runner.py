from ac_test_example_training_buffer import demo_basic
import torch
import torch.multiprocessing as mp

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    n_gpus = torch.cuda.device_count()
    print(n_gpus)
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    #original_state_dict = torch.load(filepath.format(config["load_model"]))
    #sleep(20)
    run_demo(demo_basic, world_size)

