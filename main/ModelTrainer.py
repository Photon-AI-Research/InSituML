import os

import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
import wandb

from ModelsEnum import TaskEnum
from ModelHelpers.cINN.model.modules import dataset as cinn_dataset
from ModelHelpers.cINN.model.modules import utils as cinn_utils
from ModelHelpers.DeviceHelper import DeviceDataLoader
from trainer import Trainer
from utils.dataset_utils import DIMENSION_MAP, EfieldDataset, get_tasks_datasets
from utils.plot_helper import plot_reconstructed_data, plot_reconstructed_data_taskwise, plot_heatmap_df


class ModelTrainer(Trainer):

    def __init__(
            self,
            model_path,
            model_loss_func,
            input_channels,
            number_model_layers,
            number_conv_layers,
            filters,
            latent_size,
            epochs,
            learning_rate,
            run_name,
            input_sizes,
            number_of_tasks,
            dataset_name,
            classes,
            saveModelInterval,
            model_type=None,
            e_field_dimension=None,
            is_e_field=False,
            data_path=None,
            task_enum=TaskEnum.OTHER,
            activation="leaky_relu",
            optimizer="adam",
            batch_size=3,
            onlineEWC=False,
            ewc_lambda=0.0,
            gamma=0.0,
            mas_lambda=0.0,
            agem_l_enc_lambda=1,
    ):

        super().__init__(
            model_path,
            model_loss_func,
            input_channels,
            number_model_layers,
            number_conv_layers,
            filters,
            latent_size,
            epochs,
            learning_rate,
            run_name,
            input_sizes,
            saveModelInterval,
            model_type=model_type,
            activation=activation,
            optimizer=optimizer,
            batch_size=batch_size,
            onlineEWC=onlineEWC,
            ewc_lambda=ewc_lambda,
            gamma=gamma,
            mas_lambda=mas_lambda,
            agem_l_enc_lambda=agem_l_enc_lambda,
            model_kwargs=None,
        )
        wandb.watch(self.model, log="all")
        self.train_data_sets = None
        self.test_data_sets = None
        if is_e_field:
            task_enum = TaskEnum.E_FIELD
        self.task_enum = task_enum
        self.e_field_dimension = e_field_dimension
        self.data_path = data_path
        self.number_of_tasks = number_of_tasks
        self.dataset_name = dataset_name
        self.classes = classes
        self.losses_on_first_task = [] 
        self._init_train_datasets()
        self.train_offline = True if len(self.train_data_sets) == 1 else False
        self.prev_tasks_losses = {}
        self.prev_encoded_losses = {"M 1:{}".format(i+1):[] for i in range(self.number_of_tasks)}
        self.encoded_data = []
        self.prev_tasks_images = {}
        self.last_task_included = False
        self.prev_task_ids = [i for i in range(self.number_of_tasks) if i % self.save_model_interval == 0]
        if self.number_of_tasks - 1 not in self.prev_task_ids:
            self.prev_task_ids.append(self.number_of_tasks - 1)
            self.last_task_included = True

    @property
    def is_e_field(self):
        return self.task_enum is TaskEnum.E_FIELD

    def _create_pc_datasets(self, data_path):
        run_settings = cinn_utils.load_run_settings(data_path)
        path_to_particle_data = run_settings['path_to_particle_data']
        path_to_radiation_data = run_settings['path_to_radiation_data']

        paths_to_PS = [
            path_to_particle_data + '/' + next_file
            for next_file in os.listdir(path_to_particle_data)
        ]
        paths_to_PS.sort()
        paths_to_radiation = [
            path_to_radiation_data + '/' + next_file
            for next_file in os.listdir(path_to_radiation_data)
        ]
        paths_to_radiation.sort()

        if len(paths_to_PS) == 1:
            paths_to_PS.append(paths_to_PS[0])
            paths_to_radiation.append(paths_to_radiation[0])

        dataset_tr = cinn_dataset.PCDataset(
            items_phase_space=paths_to_PS[:-1],
            items_radiation=paths_to_radiation[:-1],
            num_points=20,
            num_files=-1,
            chunk_size=int(run_settings['chunk_size']),
            species=run_settings['species'],
            normalize=True,
            a=0.0,
            b=1.0,
        )

        dataset_val = cinn_dataset.PCDataset(
            items_phase_space=paths_to_PS[-1:],
            items_radiation=paths_to_radiation[-1:],
            num_points=20,
            num_files=-1,
            chunk_size=int(run_settings['chunk_size']),
            species=run_settings['species'],
            normalize=True,
            a=0.0,
            b=1.0,
        )
        return dataset_tr, dataset_val

    def _init_train_datasets(self):
        if self.is_e_field:
            no_datasets, _ = divmod(self.classes, self.number_of_tasks)
            if no_datasets <= 0:
                raise ValueError(
                    f'`classes` ({self.classes}) must be larger than or equal '
                    f'to `--nTasks` ({self.number_of_tasks})'
                )
            train_datasets = []
            test_datasets = []
            for i in range(self.number_of_tasks):
                train_datasets.append(EfieldDataset(
                    self.data_path,
                    range((i * no_datasets) + 1, (i * no_datasets) + no_datasets + 1),
                    self.e_field_dimension,
                ))
                # test_datasets.append(EfieldDataset(
                #     self.data_path,
                #     [(i * no_datasets) + 1],
                #     self.e_field_dimension,
                # ))
            self.train_data_sets = train_datasets
            self.test_data_sets = train_datasets
        elif self.task_enum is TaskEnum.PC_FIELD:
            dataset_tr, dataset_val = self._create_pc_datasets(self.data_path)

            self.train_data_sets = [dataset_tr]
            self.test_data_sets = [dataset_val]
        else:
            self.train_data_sets = get_tasks_datasets(
                self.dataset_name,
                self.data_path,
                self.classes,
                self.number_of_tasks,
            )
            self.test_data_sets = get_tasks_datasets(
                self.dataset_name,
                self.data_path,
                self.classes,
                self.number_of_tasks,
                train=False,
            )

    def _train_offline(self):
        print("GOING OFFLINE......")
        for epoch in range(1, self.epochs + 1):
            losses = []
            for data_set in self.train_data_sets:
                data_loader = DataLoader(data_set, self.batch_size, num_workers = 2, pin_memory=True, shuffle=True)
                device_data_loader = DeviceDataLoader(data_loader,self.device)
                for batch, labels in device_data_loader:
                    if self.is_mlp:
                        batch = self._modify_batch(batch)
                    encoded , decoded, _ = self.model(batch)
                    self.optimizer.zero_grad()
                    loss = self.loss_func(batch, decoded)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
            wandb.log({
                        "loss_model": sum(losses) / len(losses),
                        "epoch": epoch
            })
            self.validate(epoch)
        self._save_model(self.run_name,self.model_path,"")
        
        self.plot_reconstructed_data_for_offline()
    
    def _train_taskwise(self):
        for task_id, data_set in enumerate(self.train_data_sets):
            task_loss = []
            list_of_encoded = []
            for e in range(self.epochs):
                losses = []
                data_loader = DataLoader(data_set, self.batch_size, num_workers = 2, pin_memory=True, shuffle=True)
                device_data_loader = DeviceDataLoader(data_loader,self.device)
                for batch, labels in device_data_loader:
                    if self.is_mlp:
                        batch = self._modify_batch(batch)
                    encoded , decoded, _ = self.model(batch)
                    list_of_encoded.append(encoded.mean(dim = 0).detach().cpu())
                    self.optimizer.zero_grad()
                    loss = self.loss_func(batch, decoded)
                    #ewc-loss-if-needed
                    if self.ewc_lambda > 0.0:
                        ewc_loss = self.ewc_lambda * self.model.ewc_loss()
                        loss += ewc_loss
                        if e == self.epochs - 1:
                            wandb.log({
                                "ewc_loss" : ewc_loss,
                                "task" : task_id
                            })

                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                loss_avg = sum(losses) / len(losses)
                task_loss.append(loss_avg)
                wandb.log({
                     "loss_model": loss_avg,
                })
            wandb.log({
                     "task_loss": sum(task_loss) / len(task_loss),
                     "task": task_id + 1
                })
            #self._reset_optimizer()
            # if task_id == 0:
            #     continue
            if self.is_e_field and task_id !=0:
                self._append_loss_on_first_task()
            
            """ 
                Estimating Fisher after every task learned
            """ 
            if self.ewc_lambda > 0.0 and task_id != self.number_of_tasks - 1:
                self.model.estimate_fisher(data_set, self.loss_func, self.is_mlp)

            if task_id % self.save_model_interval == 0:
                self.generate_data_for_img_plots_recon(task_id)  
            self.store_validate_encoded(sum(list_of_encoded) / len(list_of_encoded), task_id)
            self.validate_prev_tasks(task_id)
            self._save_model(self.run_name,self.model_path, task_id)
               
        if self.last_task_included:
            #self._save_model(self.run_name,self.model_path, self.number_of_tasks - 1)
            #self.validate_prev_tasks(self.number_of_tasks - 1)   
            self.generate_data_for_img_plots_recon(self.number_of_tasks - 1)
        self.plot_reconstructed_data_grid_task_wise()

    def _append_loss_on_first_task(self):
        loss = self._get_validation_loss_avg(self.test_data_sets[1])
        self.losses_on_first_task.append(loss["mse_loss"])

    def train(self):    
        if self.train_offline:
            self._train_offline()
        else:
            self._train_taskwise()
            self.log_prev_tasks_data()
        
        self.validate_class_wise()

    def _get_validation_loss_avg(self, dataset):
        val_losses = {
            "l1_loss":[],
            "inf_norm_loss":[],
            "mse_loss":[]
        }
        val_data_loader = DataLoader(dataset, self.batch_size, num_workers = 1, pin_memory=True, shuffle=False)
        val_device_data_loader = DeviceDataLoader(val_data_loader,self.device)
        for batch, labels in val_device_data_loader:
            if self.is_mlp:
                batch = self._modify_batch(batch)
            encoded , decoded, _ = self.model(batch)
            val_loss = self.loss_func(batch, decoded)
            val_losses["mse_loss"].append(val_loss.item())

            val_loss = self.l1_loss(batch, decoded)
            val_losses["l1_loss"].append(val_loss.item())

            val_loss = self._infinity_norm_loss(batch, decoded)
            val_losses["inf_norm_loss"].append(val_loss.item())
        val_losses = {n: sum(p)/len(p) for n, p in val_losses.items()}
        return val_losses

    def validate_prev_tasks(self, task_id):
        self.model.eval()
        with torch.no_grad():
            #idx = self.prev_task_ids.index(task_id)
            for id in range(task_id + 1):
                #prev_idx = self.prev_task_ids[id]
                val_data_set = self.test_data_sets[id]
                avg_val_loss = self._get_validation_loss_avg(val_data_set)
                if "M 1:{}".format(task_id+1) in self.prev_tasks_losses.keys():
                    self.prev_tasks_losses["M 1:{}".format(task_id+1)].append(avg_val_loss["mse_loss"])
                else:
                    self.prev_tasks_losses["M 1:{}".format(task_id+1)] = [avg_val_loss["mse_loss"]]
        self.model.train()
    
    def store_validate_encoded(self,encoded, task_id):
        self.encoded_data.append(encoded)
        for i in range(task_id):
            enc_loss = self.loss_func(self.encoded_data[i],encoded)
            self.prev_encoded_losses["M 1:{}".format(task_id+1)].append(enc_loss.item())
        self.prev_encoded_losses["M 1:{}".format(task_id+1)].append(0.0)

    def validate(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for data_set in self.test_data_sets:
                avg_val_loss = self._get_validation_loss_avg(data_set)
                wandb.log({
                            "validation_loss_model": avg_val_loss["mse_loss"],
                            "epoch": epoch
                })
        self.model.train()
    
    def validate_class_wise(self):
        if self.is_e_field:
            data_sets = []
            for i in range(self.classes):
                data_sets.append(EfieldDataset(
                    self.data_path, [i+1], self.e_field_dimension))
        else:
            data_sets = get_tasks_datasets(
                self.dataset_name,
                self.data_path,
                self.classes,
                num_tasks=self.classes,
                train=False,
            )
        self.model.eval()
        with torch.no_grad():
            for task_id, data_set in enumerate(data_sets):
                avg_val_loss = self._get_validation_loss_avg(data_set)
                wandb.log({
                            "validation_MSE_loss_per_class": avg_val_loss["mse_loss"],
                            "validation_L1_loss_per_class": avg_val_loss["l1_loss"],
                            "validation_InfinityNorm_loss_per_class": avg_val_loss["inf_norm_loss"],
                            "class": task_id + 1
                })
        self.model.train()

    def log_prev_tasks_data(self):
        plot_df = pd.DataFrame.from_dict(self.prev_tasks_losses, orient='index')
        plot_df.columns = ["Task {}".format(i + 1) for i in range(self.number_of_tasks)]

        enc_df = pd.DataFrame.from_dict(self.prev_encoded_losses,orient='index')
        enc_df.columns = ["Task {}".format(i + 1) for i in range(self.number_of_tasks)]
        
        wandb.log(
                {
                    "l2 norm task wise":plot_df,
                    "encoded l2 loss": enc_df

                })

        plot_df = plot_df.T
        if not self.is_e_field:
            loss_on_first_task = plot_df.loc["Task 1"].tolist()
            for idx,loss in enumerate(loss_on_first_task):
                wandb.log(
                {
                    "loss_on_first_task":loss,
                    "task": idx + 1
                })
        else:
            for idx,loss in enumerate(self.losses_on_first_task):
                wandb.log(
                {
                    "loss_on_first_task":loss,
                    "task": idx + 2
                })

        wandb.log({
            "Prev Tasks Validation Losses": wandb.Image(plot_heatmap_df(plot_df.T, "Task-wise L2-Loss", normalize = True)),
            "Prev Tasks Encoded Losses": wandb.Image(plot_heatmap_df(enc_df, "Encoded L2-Loss", normalize = True))
            #plot_df.plot(kind = "bar", figsize = (5*len(self.prev_task_ids),20)).legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        })

    def generate_data_for_img_plots_recon(self,task_id):
        self.model.eval()
        with torch.no_grad():
            idx = self.prev_task_ids.index(task_id)
            reconstructed = []
            og = []
            last_one = False
            if idx == len(self.prev_task_ids) - 1:
                last_one = True
            for id in range(idx + 1):
                prev_idx = self.prev_task_ids[id]
                val_data_set = self.test_data_sets[prev_idx]
                val_data_loader = DataLoader(val_data_set, 1, num_workers = 1, pin_memory=True, shuffle=False)
                val_device_data_loader = DeviceDataLoader(val_data_loader,self.device)
                for batch, label in val_device_data_loader:
                    if last_one:
                        og.append(batch)
                    if self.is_mlp:
                        batch = self._modify_batch(batch)
                    encoded , decoded, _ = self.model(batch)
                    if self.is_mlp:
                        decoded = self._modify_batch(decoded, reverse = True)
                    reconstructed.append(decoded)
                    break
            reconstructed_imgs = self.prepare_data_to_plot(reconstructed)
            self.prev_tasks_images["M 1:{}".format(task_id+1)] = reconstructed_imgs
            if last_one:
                og_imgs = self.prepare_data_to_plot(og)
                self.prev_tasks_images["og_imgs"] = og_imgs
        self.model.train()

    def plot_reconstructed_data_for_offline(self):
        if self.is_e_field:
            data_sets = self.test_data_sets
        else:
            data_sets = get_tasks_datasets(
                self.dataset_name,
                self.data_path,
                self.classes,
                num_tasks=self.classes,
                train=False,
            )

        self.model.eval()
        original = []
        reconstructed = []
        count = 0
        with torch.no_grad():
            for data_set in data_sets:
                val_data_loader = DataLoader(data_set, 1, num_workers = 1, pin_memory=True, shuffle=True)
                val_device_data_loader = DeviceDataLoader(val_data_loader,self.device)
                for i in range(5):
                    batch, label = val_device_data_loader[i]
                    original.append(batch)
                    if self.is_mlp:
                        batch = self._modify_batch(batch)
                    encoded , decoded, _ = self.model(batch)
                    if self.is_mlp:
                        decoded = self._modify_batch(decoded, reverse = True)
                    reconstructed.append(decoded)
        
        original, reconstructed = self.prepare_data_to_plot(original), self.prepare_data_to_plot(reconstructed)
        plot_figure = plot_reconstructed_data(original, reconstructed, "Precitions", cmap = 'jet' if self.is_e_field else 'gray')
        wandb.log({"Reconstructed Images":plot_figure})
        
        self.model.train()

    def prepare_data_to_plot(self, img_data):
        og = []
        for i in range(len(img_data)):
            if self.is_e_field:
                if self.e_field_dimension is not None:
                    idx = DIMENSION_MAP[self.e_field_dimension]
                    og.append(img_data[i].detach().cpu()[0][idx][32].numpy())
                else:
                    og.append(img_data[i].detach().cpu()[0][1][32].numpy())
            else:
                og.append(img_data[i].detach().cpu()[0].permute(1, 2, 0).numpy())
        
        return og
    
    def plot_reconstructed_data_grid_task_wise(self):
        fig_plot = plot_reconstructed_data_taskwise(self.prev_tasks_images["og_imgs"], self.prev_tasks_images, self.prev_task_ids, "Reconstruction Grid", cmap = 'jet' if self.is_e_field else 'gray')
        wandb.log({"Reconstructed Imgs Grid": fig_plot})

