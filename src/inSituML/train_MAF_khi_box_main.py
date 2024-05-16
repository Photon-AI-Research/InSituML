import os
import numpy as np
import torch
from ks_models import PC_MAF
import torch.optim as optim
import time
import wandb
from utilities import normalize_point, denormalize_point


def sample_pointcloud(model, num_samples, cond, vmin, vmax):
    model.model.eval()
    with torch.no_grad():
        # pc_pr = (model.model.sample(10, cond)).squeeze(1)
        pc_pr = model.model.sample(num_samples, cond)
        # pc_pr = data_preprocessing.denormalize_point(pc_pr.to('cpu'),
        #                            self.vmin_ps, self.vmax_ps, self.a, self.b)

        print(pc_pr.shape)

        # print(pc_pr.shape[0])
        # Calculate the number of rows needed in the reshaped array
        num_rows = pc_pr.shape[0] * pc_pr.shape[1]
        num_columns = pc_pr.shape[-1]

        # Reshape the tensor
        pc_pr_reshaped = pc_pr.view(num_rows, num_columns)

        pc_pr_reshaped = denormalize_point(
            pc_pr_reshaped.cpu(), vmin, vmax, a=0.0, b=1.0
        )
        pc_pr = pc_pr_reshaped.reshape(pc_pr.shape)
    return pc_pr


def generate_one_hot_tensors(num_categories, num_samples):
    one_hot_tensors_sequence = []

    for i in range(num_samples):
        one_hot_tensor = torch.zeros(num_categories)
        one_hot_tensor[i % num_categories] = 1
        one_hot_tensors_sequence.append(one_hot_tensor)

    one_hot_tensors_sequence = torch.stack(one_hot_tensors_sequence)

    return one_hot_tensors_sequence


def save_checkpoint(
    model, optimizer, path, last_loss, min_valid_loss, epoch, wandb_run_id
):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "last_loss": last_loss.item(),
        "epoch": epoch,
        "min_valid_loss": min_valid_loss,
        "wandb_run_id": wandb_run_id,
    }

    torch.save(state, path + "/model_" + str(epoch))


class Loader:
    def __init__(
        self,
        pathpattern="/data_box/60_70_130_140_0_12/{}.npy",
        t0=0,
        t1=100,
        timebatchsize=20,
        particlebatchsize=10240,
    ):
        self.pathpattern = pathpattern

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
                self, loader, t0, t1, timebatchsize=20, particlebatchsize=10240
            ):
                self.perm = torch.randperm(len(loader))
                self.loader = loader
                self.t0 = t0
                self.t1 = t1
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
                    index = time_ + self.t0
                    p = torch.from_numpy(
                        np.load(self.loader.pathpattern.format(index)).astype(
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

        return Epoch(
            self, self.t0, self.t1, self.timebatchsize, self.particlebatchsize
        )


if __name__ == "__main__":

    hyperparameter_defaults = dict(
        t0=1999,
        t1=2001,
        timebatchsize=2,
        particlebatchsize=1024,
        dim_condition=2,
        num_coupling_layers=6,
        hidden_size=256,
        lr=0.00001,
        num_epochs=20000,
        num_blocks_mat=6,
        activation="relu",
        pathpattern="60-70-130-140-0-12",
    )

    print("New session...")
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults, project="khi_public")
    start_epoch = 0
    min_valid_loss = np.inf

    # Access all hyperparameter values through wandb.config
    config = wandb.config

    file_path = "data_box/"

    # min/max of particle dataa for normalisation
    # pos_minmax = np.load('/bigdata/hplsim/aipp/Jeyhun/khi/pos_minmax.npy')
    # Split the values using '_' as a separator
    print(config["pathpattern"], 'config["pathpattern"]')
    values = config["pathpattern"].split("-")

    # Convert the values to floats
    xmin, xmax, ymin, ymax, zmin, zmax = map(float, values)

    # Create a numpy array
    pos_minmax = np.array([[xmin, ymin, zmin], [xmax, ymax, zmax]])
    print("pos_minmax", pos_minmax)

    pathpattern = (
        file_path + config["pathpattern"].replace("-", "_") + "/{}.npy"
    )

    loader_ = Loader(
        pathpattern=pathpattern,
        t0=config["t0"],
        t1=config["t1"],
        timebatchsize=config["timebatchsize"],
        particlebatchsize=config["particlebatchsize"],
    )

    model = PC_MAF(
        dim_condition=config["dim_condition"],
        dim_input=9,
        num_coupling_layers=config["num_coupling_layers"],
        hidden_size=config["hidden_size"],
        device="cuda",
        weight_particles=False,
        num_blocks_mat=config["num_blocks_mat"],
        activation=config["activation"],
    )

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.8
    )

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
    for i_epoch in range(start_epoch, config["num_epochs"]):
        print("i_epoch:", i_epoch)
        loss_overall = []
        for tb in range(len(epoch)):
            loss_avg = []
            # print('tb:', tb)
            timebatch = epoch[tb]

            start_timebatch = time.time()
            for b in range(len(timebatch)):
                optimizer.zero_grad()

                phase_space, times = timebatch[b]
                phase_space = normalize_point(
                    phase_space, pos_minmax[0], pos_minmax[1], a=0.0, b=1.0
                )
                loss = -model.model.log_prob(
                    inputs=phase_space.to(model.device),
                    context=times.to(model.device),
                )
                loss = loss.mean()
                loss_avg.append(loss.item())
                loss.backward()
                optimizer.step()

            end_timebatch = time.time()
            elapsed_timebatch = end_timebatch - start_timebatch

            loss_timebatch_avg = sum(loss_avg) / len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            print(
                ("i_epoch:{}, tb: {}, last timebatch loss: {}, avg_loss: {}, " +
                 "time: {}").format(
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
                f"Validation Loss Decreased({min_valid_loss:.6f}--->"
                + f"{loss_overall_avg:.6f}) \t Saving The Model"
            )
            min_valid_loss = loss_overall_avg
            # Saving State Dict
            torch.save(
                model.state_dict(),
                directory + "/best_model_",
                _use_new_zipfile_serialization=False,
            )
        if (i_epoch + 1) % 100 == 0:
            print("Saving a checkpoint...")
            save_checkpoint(
                model,
                optimizer,
                directory,
                loss,
                min_valid_loss,
                i_epoch,
                wandb.run.id,
            )

        scheduler.step()

        # Log the loss and accuracy values at the end of each epoch
        wandb.log(
            {
                # "Epoch": i_epoch,
                # "last time batch loss":loss.item(),
                "loss_timebatch_avg_loss": loss_timebatch_avg,
                "loss_overall_avg": loss_overall_avg,
                "min_valid_loss": min_valid_loss,
            }
        )

    # Code or process to be measured goes here
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.6f} seconds")
