# _v1_5_1
import os
import numpy as np
import torch
from ks_models import PC_MAF as model_MAF
import torch.optim as optim
import time
import wandb
import sys
<<<<<<< HEAD


def normalize_point(point, vmin, vmax, a=0.0, b=1.0):
    """
    Normalize point from a set of points with vmin(minimum) and vmax(maximum)
    to be in a range [a, b]
    """

    # Extract the first three columns
    first_three_col = point[:, :3]

    # Perform operations on the first three columns
    modified_first_three_col = a + (first_three_col - vmin) * (b - a) / (
        vmax - vmin
    )

    # Combine the modified columns with the unchanged columns
    result_array = torch.cat(
        (modified_first_three_col, point[:, 3:]), dim=1
    ).to(point.dtype)

    return result_array


def denormalize_point(point_normalized, vmin, vmax, a=0.0, b=1.0):
    """
    Denormalize point back to the original range using vmin(minimum)
         and vmax(maximum).
    """

    # Convert the input to PyTorch tensors
    # point_normalized = torch.tensor(point_normalized)
    vmin = torch.tensor(vmin)
    vmax = torch.tensor(vmax)

    # Extract the first three columns
    first_three_col_normalized = point_normalized[:, :3]

    # Perform operations on the first three columns to denormalize them
    denormalized_first_three_col = vmin + (first_three_col_normalized - a) * (
        vmax - vmin
    ) / (b - a)

    # Combine the denormalized columns with the unchanged columns
    result_array = torch.cat(
        (denormalized_first_three_col, point_normalized[:, 3:]), dim=1
    ).to(point_normalized.dtype)

    return result_array
=======
from utilities import normalize_point, denormalize_point
>>>>>>> a300c41 (Move normalize/denormalize functions to utilities.py)


def save_checkpoint(
    model, optimizer, path, last_loss, min_valid_loss, epoch, wandb_run_id
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "last_loss": loss.item(),
        "epoch": epoch,
        "min_valid_loss": min_valid_loss,
        "wandb_run_id": wandb_run_id,
    }

    torch.save(state, path + "/model_" + str(epoch))


class Loader:
    def __init__(
        self,
        pathpattern="/bigdata/hplsim/aipp/Jeyhun/khi/particle/{}.npy",
        t0=0,
        t1=100,
        timebatchsize=20,
        particlebatchsize=10240,
    ):
        self.pathpattern = pathpattern

        # check if all files are there

        self.t0 = t0
        self.t1 = t1

        self.timebatchsize = timebatchsize
        self.particlebatchsize = particlebatchsize

        num_files = t1 - t0
        missing_files = [
            i
            for i in range(t0, t1)
            if not os.path.exists(pathpattern.format(i))
        ]
        num_missing = len(missing_files)
        all_files_exist = num_missing == 0

        if all_files_exist:
            print(
                "All {} files from {} to {} exist in the directory.".format(
                    num_files, t0, t1
                )
            )
        else:
            print(
                "{} files are missing out of {} in the directory.".format(
                    num_missing, num_files
                )
            )

    def __len__(self):
        return self.t1 - self.t0

    def __getitem__(self, idx):

        class Epoch:
            def __init__(
                self, loader, timebatchsize=20, particlebatchsize=10240
            ):
                self.perm = torch.randperm(len(loader))
                self.loader = loader
                self.timebatchsize = timebatchsize
                self.particlebatchsize = particlebatchsize

            def __len__(self):
                return len(self.loader) // self.timebatchsize

            def __getitem__(self, timebatch):
                i = self.timebatchsize * timebatch
                bi = self.perm[i:i + self.timebatchsize]
                times = []
                particles = []
                for time_ in bi:
                    p = torch.from_numpy(
                        np.load(self.loader.pathpattern.format(time_)).astype(
                            np.float32
                        )
                    )
                    particles.append(p)
                    t = torch.zeros((p.shape[0], len(self.loader)))
                    t[:, time_] = 1
                    times.append(t)

                particles = torch.cat(particles)
                times = torch.cat(times)

                class Timebatch:
                    def __init__(self, particles, times, batchsize):
                        self.batchsize = batchsize
                        self.particles = particles
                        self.times = times

                        self.perm = torch.randperm(self.times.shape[0])

                    def __len__(self):
                        return self.times.shape[0] // self.batchsize

                    def __getitem__(self, batch):
                        i = self.batchsize * batch
                        bi = self.perm[i:i + self.batchsize]

                        return self.particles[bi], self.times[bi]

                return Timebatch(particles, times, self.particlebatchsize)

        return Epoch(self, self.timebatchsize, self.particlebatchsize)


if __name__ == "__main__":

    hyperparameter_defaults = dict(
        t0=100,
        t1=110,
        timebatchsize=4,
        particlebatchsize=10240,
        dim_condition=10,
        num_coupling_layers=5,
        hidden_size=256,
        lr=0.00001,
        num_epochs=2000,
    )

    # min/max of particle dataa for normalisation
    pos_minmax = np.load("/bigdata/hplsim/aipp/Jeyhun/khi/pos_minmax.npy")

    loader_ = Loader(
        t0=hyperparameter_defaults["t0"],
        t1=hyperparameter_defaults["t1"],
        timebatchsize=hyperparameter_defaults["timebatchsize"],
        particlebatchsize=hyperparameter_defaults["particlebatchsize"],
    )

    model = model_MAF.PC_MAF(
        dim_condition=hyperparameter_defaults["dim_condition"],
        dim_input=9,
        num_coupling_layers=hyperparameter_defaults["num_coupling_layers"],
        hidden_size=hyperparameter_defaults["hidden_size"],
        device="cuda",
        enable_wandb=False,
        weight_particles=False,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=hyperparameter_defaults["lr"]
    )

    arg1 = sys.argv[1]
    if arg1 == "checkpoint":
        print("Checkpoint session loaded...")
        checkpoint_path = (
            "/bigdata/hplsim/aipp/Jeyhun/khi/checkpoints/6d7p03r1/model_1052"
        )
        checkpoint = torch.load(checkpoint_path)
        last_loss = checkpoint["last_loss"]
        wandb_run_id = checkpoint["wandb_run_id"]

        start_epoch = checkpoint["epoch"]
        min_valid_loss = checkpoint["min_valid_loss"]

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Pass your defaults to wandb.init
        wandb.init(
            config=hyperparameter_defaults,
            project="khi_public",
            resume=wandb_run_id,
        )
    else:
        print("New session...")
        # Pass your defaults to wandb.init
        wandb.init(config=hyperparameter_defaults, project="khi_public")
        start_epoch = 0
        min_valid_loss = np.inf

    # Access all hyperparameter values through wandb.config
    config = wandb.config

    # path = '/bigdata/hplsim/aipp/Jeyhun/khi/checkpoints/khi_'
    # + str(wandb.run.id) + model.num_coupling_layers + st

    directory = "/bigdata/hplsim/aipp/Jeyhun/khi/checkpoints/" + str(
        wandb.run.id
    )

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    epoch = loader_[0]
    start_time = time.time()
    # num_epochs = 1
    for i_epoch in range(start_epoch, hyperparameter_defaults["num_epochs"]):
        print("i_epoch:", i_epoch)
        loss_overall = []
        for tb in range(len(epoch)):
            loss_avg = []
            # print('tb:', tb)
            timebatch = epoch[tb]

            start_timebatch = time.time()
            for b in range(len(timebatch)):
                optimizer.zero_grad()
                # print('b',b)
                phase_space, times = timebatch[b]

                phase_space = normalize_point(
                    phase_space, pos_minmax[0], pos_minmax[1], a=0.0, b=1.0
                )

                # for param in model.model.parameters():
                #     param.grad = None

                loss = -model.model.log_prob(
                    inputs=phase_space.to(model.device),
                    context=times.to(model.device),
                )
                # if model.weight_particles:
                #     loss = loss*phase_space[:,-1]
                loss = loss.mean()
                loss_avg.append(loss.item())
                loss.backward()
                optimizer.step()

            end_timebatch = time.time()
            elapsed_timebatch = end_timebatch - start_timebatch

            loss_timebatch_avg = sum(loss_avg) / len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            print(
                ("i_epoch:{}, tb: {}, last timebatch " +
                 "loss: {}, avg_loss: {}, time: {}").format(
                    i_epoch,
                    tb,
                    loss.item(),
                    loss_timebatch_avg,
                    elapsed_timebatch,
                )
            )

        loss_overall_avg = sum(loss_overall) / len(loss_overall)

        if min_valid_loss > loss_overall_avg:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->" +
                f"{loss_overall_avg:.6f}) \t Saving The Model"
            )
            min_valid_loss = loss_overall_avg
            # Saving State Dict
            torch.save(
                model.state_dict(),
                directory + "/best_model_",
                _use_new_zipfile_serialization=False,
            )

        # Log the loss and accuracy values at the end of each epoch
        wandb.log(
            {
                "Epoch": i_epoch,
                "last time batch loss": loss.item(),
                "loss_timebatch_avg_loss": loss_timebatch_avg,
                "loss_overall_avg": loss_overall_avg,
                "min_valid_loss": min_valid_loss,
            }
        )

        save_checkpoint(
            model,
            optimizer,
            directory,
            loss,
            min_valid_loss,
            i_epoch,
            wandb.run.id,
        )

    # Code or process to be measured goes here
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.6f} seconds")
