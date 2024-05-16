# train_AE_khi_box_v2-2-1-8-rad-ex-random-v4-chamfers-Copy1.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import wandb
import utilities


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


def chamfersDist(a, b):
    d = torch.cdist(a, b, p=2)
    return torch.sum(torch.min(d, -1).values) + torch.sum(
        torch.min(d, -2).values
    )


class TrainLoader:
    def __init__(
        self,
        pathpattern1=("/bigdata/hplsim/aipp/Jeyhun/khi/particle_box/" +
                      "40_80_80_160_0_2/{}.npy"),
        pathpattern2=("/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/" +
                      "radiation_ex/{}.npy"),
        t0=0,
        t1=100,
        timebatchsize=20,
        particlebatchsize=10240,
        blacklist_box=None,
    ):

        self.pathpattern1 = pathpattern1
        self.pathpattern2 = pathpattern2
        self.blacklist_box = blacklist_box

        # TODO check if all files are there

        self.t0 = t0
        self.t1 = t1

        self.timebatchsize = timebatchsize
        self.particlebatchsize = particlebatchsize

        num_files = t1 - t0
        missing_files = [
            i
            for i in range(t0, t1)
            if not os.path.exists(pathpattern1.format(i))
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
                self,
                loader,
                t0,
                t1,
                timebatchsize=20,
                particlebatchsize=10240,
                blacklist_box=None,
            ):

                self.perm = torch.randperm(len(loader))
                self.loader = loader
                self.t0 = t0
                self.t1 = t1
                self.timebatchsize = timebatchsize
                self.particlebatchsize = particlebatchsize
                self.blacklist_box = blacklist_box

            def __len__(self):
                return len(self.loader) // self.timebatchsize

            def __getitem__(self, timebatch):
                i = self.timebatchsize * timebatch
                bi = self.perm[i:i + self.timebatchsize]
                radiation = []
                particles = []
                for time_ in bi:
                    index = time_ + self.t0

                    # Load particle data
                    p = np.load(
                        self.loader.pathpattern1.format(index),
                        allow_pickle=True,
                    )

                    # Load radiation data
                    r = torch.from_numpy(
                        np.load(self.loader.pathpattern2.format(index)).astype(
                            np.cfloat
                        )
                    )

                    # Exclude the blacklisted box if specified
                    if self.blacklist_box is not None:
                        p = np.delete(p, self.blacklist_box, axis=0)
                        r = np.delete(r, self.blacklist_box, axis=0)

                    # Normalize particle data for all boxes
                    p = [utilities.normalize_columns(element) for element in p]
                    p = np.array(p, dtype=object)

                    # random sample N points from each box
                    p = [
                        utilities.random_sample(element, sample_size=150000)
                        for element in p
                    ]
                    p = torch.from_numpy(np.array(p, dtype=np.float32))

                    # choose relevant directions
                    r = r[:, 1:, :]
                    r = r.view(r.shape[0], -1)

                    # Compute the phase (angle) of the complex number in radians
                    phase = torch.angle(r)

                    # Compute the amplitude (magnitude) of the complex number
                    amplitude = torch.abs(r)
                    r = torch.cat((amplitude, phase), dim=1).to(torch.float32)

                    particles.append(p)
                    radiation.append(r)

                # concatenate particle and radiation data across randomly
                # chosen timesteps
                particles = torch.cat(particles)
                radiation = torch.cat(radiation)

                class Timebatch:
                    def __init__(self, particles, radiation, batchsize):
                        self.batchsize = batchsize
                        self.particles = particles
                        self.radiation = radiation

                        self.perm = torch.randperm(self.radiation.shape[0])

                    def __len__(self):
                        return self.radiation.shape[0] // self.batchsize

                    def __getitem__(self, batch):
                        i = self.batchsize * batch
                        bi = self.perm[i:i + self.batchsize]

                        return self.particles[bi], self.radiation[bi]

                return Timebatch(particles, radiation, self.particlebatchsize)

        return Epoch(
            self,
            self.t0,
            self.t1,
            self.timebatchsize,
            self.particlebatchsize,
            self.blacklist_box,
        )


class ValidationDataset:
    def __init__(
        self,
        pathpattern1,
        pathpattern2,
        validation_boxes,
        t0=0,
        t1=100,
        timebatchsize=20,
        particlebatchsize=10240,
    ):
        self.pathpattern1 = pathpattern1
        self.pathpattern2 = pathpattern2
        self.validation_boxes = validation_boxes
        self.t0 = t0
        self.t1 = t1
        self.timebatchsize = timebatchsize
        self.particlebatchsize = particlebatchsize

    def __len__(self):
        return self.t1 - self.t0

    def __getitem__(self, idx):

        timestep_index = self.t0 + idx

        # Load particle data for the validation boxes
        p_loaded = np.load(
            self.pathpattern1.format(timestep_index), allow_pickle=True
        )
        p = [p_loaded[box_index] for box_index in self.validation_boxes]

        p = [utilities.normalize_columns(element) for element in p]
        p = np.array(p, dtype=object)

        p = [utilities.random_sample(element, sample_size=150000)
             for element in p]
        p = torch.from_numpy(np.array(p, dtype=np.float32))

        # Load radiation data for the validation boxes
        r_loaded = torch.from_numpy(
            np.load(self.pathpattern2.format(timestep_index)).astype(np.cfloat)
        )
        r = r_loaded[self.validation_boxes, 1:, :]
        r = r.view(r.shape[0], -1)

        # Compute the phase (angle) of the complex number in radians
        phase = torch.angle(r)

        # Compute the amplitude (magnitude) of the complex number
        amplitude = torch.abs(r)
        r = torch.cat((amplitude, phase), dim=1).to(torch.float32)

        return timestep_index, self.validation_boxes, p, r


if __name__ == "__main__":

    hyperparameter_defaults = dict(
        t0=1990,
        t1=2001,
        timebatchsize=4,
        particlebatchsize=4,
        hidden_size=1024,
        dim_pool=1,
        lr=0.001,
        num_epochs=20000,
        val_boxes=[19, 5, 3],
        activation="relu",
        pathpattern1=("/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/" +
                      "particle_002/{}.npy"),
        pathpattern2=("/bigdata/hplsim/aipp/Jeyhun/khi/part_rad/" +
                      "radiation_ex_002/{}.npy"),
    )

    print("New session...")
    # Pass your defaults to wandb.init
    wandb.init(config=hyperparameter_defaults, project="khi_public")
    start_epoch = 0
    min_valid_loss = np.inf

    # Access all hyperparameter values through wandb.config
    config = wandb.config

    pathpattern1 = config["pathpattern1"]
    pathpattern2 = config["pathpattern2"]

    loader_ = TrainLoader(
        pathpattern1=pathpattern1,
        pathpattern2=pathpattern2,
        t0=config["t0"],
        t1=config["t1"],
        timebatchsize=config["timebatchsize"],
        particlebatchsize=config["particlebatchsize"],
        blacklist_box=config["val_boxes"],
    )

    validation_dataset = ValidationDataset(
        pathpattern1,
        pathpattern2,
        config["val_boxes"],
        t0=config["t0"],
        t1=config["t1"],
    )

    class Reshape(nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.shape = args

        def forward(self, x):
            return x.view(self.shape)

    # Define the convolutional autoencoder class
    class ConvAutoencoder(nn.Module):
        def __init__(self):
            super(ConvAutoencoder, self).__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv1d(9, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(256, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(512, config["hidden_size"], kernel_size=1),
                nn.AdaptiveMaxPool1d(config["dim_pool"]),
                nn.Flatten(),
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.Unflatten(1, (16, 4, 4, 4)),
                nn.ConvTranspose3d(16, 4, kernel_size=2, stride=1),
                nn.ReLU(),
                nn.ConvTranspose3d(4, 9, kernel_size=2, stride=2),
                nn.Flatten(2),
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    # Initialize the convolutional autoencoder
    model = ConvAutoencoder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1000, gamma=0.9
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

    patience = 20
    slow_improvement_patience = 10
    no_improvement_count = 0
    slow_improvement_count = 0
    batch_tot_idx = 0

    start_time = time.time()
    for i_epoch in range(start_epoch, config["num_epochs"]):
        loss_overall = []
        for tb in range(len(epoch)):
            loss_avg = []
            timebatch = epoch[tb]

            batch_idx = 0
            start_timebatch = time.time()
            for b in range(len(timebatch)):
                batch_idx += 1
                batch_tot_idx += 1
                optimizer.zero_grad()
                phase_space, _ = timebatch[b]

                phase_space = phase_space.permute(0, 2, 1).to(device)

                output = model(phase_space)

                loss = chamfersDist(
                    output.transpose(2, 1), phase_space.transpose(2, 1)
                )

                loss = loss.mean()
                loss_avg.append(loss.item())

                # Log batch loss to wandb
                wandb.log(
                    {
                        "Batch": batch_tot_idx,
                        "Batch Loss": loss.item(),
                    }
                )

                loss.backward()
                optimizer.step()

            end_timebatch = time.time()
            elapsed_timebatch = end_timebatch - start_timebatch

            loss_timebatch_avg = sum(loss_avg) / len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            print(
                ("i_epoch:{}, tb: {}, last timebatch loss: " +
                 "{}, avg_loss: {}, time: {}").format(
                    i_epoch,
                    tb,
                    loss.item(),
                    loss_timebatch_avg,
                    elapsed_timebatch,
                )
            )

        loss_overall_avg = sum(loss_overall) / len(loss_overall)

        # Calculate validation loss
        model.eval()
        val_loss_avg = []
        with torch.no_grad():
            for idx in range(len(validation_dataset)):
                timestep_index, validation_boxes, p, _ = validation_dataset[
                    idx
                ]

                p = p.permute(0, 2, 1).to(device)
                pc_pr_t = model(p)

                val_loss = chamfersDist(
                    pc_pr_t.transpose(2, 1), p.transpose(2, 1)
                )
                # emd = emd_loss(pc_pr_t.transpose(2,1).contiguous(),
                # p.transpose(2,1).contiguous())
                val_loss_avg.append(val_loss.mean().item())
            val_loss_overall_avg = sum(val_loss_avg) / len(val_loss_avg)

        if min_valid_loss > val_loss_overall_avg:
            print(
                f"Validation Loss Decreased({min_valid_loss:.6f}--->" +
                f"{val_loss_overall_avg:.6f}) \t Saving The Model"
            )
            min_valid_loss = val_loss_overall_avg
            no_improvement_count = 0
            slow_improvement_count = 0
            # Saving State Dict
            torch.save(
                model.state_dict(),
                directory + "/best_model_",
                _use_new_zipfile_serialization=False,
            )
        else:
            no_improvement_count += 1
            if (
                loss_overall_avg - min_valid_loss <= 0.001
            ):  # Adjust this threshold as needed
                slow_improvement_count += 1

        scheduler.step()

        # Log the loss and accuracy values at the end of each epoch
        wandb.log(
            {
                "Epoch": i_epoch,
                "loss_timebatch_avg_loss": loss_timebatch_avg,
                "loss_overall_avg": loss_overall_avg,
                "min_valid_loss": min_valid_loss,
            }
        )

        # if no_improvement_count >= patience or
        # slow_improvement_count >= slow_improvement_patience:
        #     break  # Stop training

    # Code or process to be measured goes here
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.6f} seconds")

    # Plotting results
#     create_position_density_plots(x, y, z, x_pr, y_pr,
#                                   z_pr, bins=100, t=t_index)

#     create_momentum_density_plots(px, py, pz, px_pr, py_pr,
#                                   pz_pr, bins=100, t=t_index)

#     create_force_density_plots(fx, fy, fz, fx_pr, fy_pr,
#                                fz_pr, bins=100, t=t_index)
