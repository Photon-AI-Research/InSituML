from torch.utils.data.dataloader import DataLoader
import wandb
from ModelHelpers.DeviceHelper import DeviceDataLoader
from utils.EpisodicMemory import EpisodicMemory
from ModelTrainer import ModelTrainer


class ReplayTrainer(ModelTrainer, EpisodicMemory):

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
            replayer_mem_size,
            aGEM_selection_size,
            store_encoded,
            layerWise=False,
            model_type=None,
            e_field_dimension=None,
            is_e_field=False,
            activation="leaky_relu",
            optimizer="adam",
            batch_size=3,
            onlineEWC=False,
            ewc_lambda=0,
            gamma=0,
            mas_lambda=0,
            agem_l_enc_lambda=1,
    ):
        ModelTrainer.__init__(
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
            model_type=model_type,
            e_field_dimension=e_field_dimension,
            is_e_field=is_e_field,
            activation=activation,
            optimizer=optimizer,
            batch_size=batch_size,
            onlineEWC=onlineEWC,
            ewc_lambda=ewc_lambda,
            gamma=gamma,
            mas_lambda=mas_lambda,
            agem_l_enc_lambda=agem_l_enc_lambda,
        )
        EpisodicMemory.__init__(self, total_mem_size = replayer_mem_size ,total_tasks= self.number_of_tasks, gradient_ref_data_size= aGEM_selection_size)
        self.store_encoded = store_encoded
        self.layerwise = layerWise

    def train_with_replay(self):
        for task_id, data_set in enumerate(self.train_data_sets):
            task_loss = []
            list_of_encoded = []
            replay_loss = []
            for _ in range(self.epochs):
                losses = []
                data_loader = DataLoader(data_set, self.batch_size, num_workers = 2, pin_memory=True, shuffle=True)
                device_data_loader = DeviceDataLoader(data_loader,self.device)
                for batch, labels in device_data_loader:
                    if self.is_mlp:
                        # need to modify so as to fit the model in-case of MLP 
                        batch = self._modify_batch(batch)
                    encoded , decoded, _ = self.model(batch)
                    list_of_encoded.append(encoded.mean(dim = 0).detach().cpu())
                    self.optimizer.zero_grad()

                    # adding tasks to replay memory as accessed except for the last one
                    if task_id != self.number_of_tasks - 1:
                        if self.is_e_field:
                            self.add_data_e_field(task_id, batch if not self.store_encoded else encoded.detach().clone(), labels)
                        else:
                            self.add_data_for_task(task_id, batch if not self.store_encoded else encoded.detach().clone(), labels)
                    
                    # sampling from replay memory and gathering its gradients
                    if task_id != 0:
                        if self.layerwise:
                            grad_loss = self.model.calculate_ref_gradients_layerwise(self.get_data_for_reference_gradient(task_id), self.loss_func, self.batch_size, self.store_encoded, loss_lambda = self.agem_l_enc_lambda)
                        else:
                            grad_loss = self.model.calculate_ref_gradients(self.get_data_for_reference_gradient(task_id), self.loss_func, self.batch_size, self.store_encoded)
                    loss = self.loss_func(batch, decoded)
                    loss.backward()

                    # overwriting gradients
                    if task_id != 0:
                        if self.layerwise:
                            self.model.overwrite_grad_layerwise()
                        else:
                            self.model.overwrite_grad()
                    self.optimizer.step()
                    losses.append(loss.item())
                    if task_id != 0:
                        replay_loss.append(grad_loss.item())

                loss_avg = sum(losses) / len(losses)
                task_loss.append(loss_avg)
                wandb.log({
                     "loss_model": loss_avg,
                })
            
            wandb.log({
                     "gradient_loss": sum(replay_loss) / len(replay_loss) if task_id != 0 else 0.0,
                     "task": task_id + 1
                })

            wandb.log({
                     "task_loss": sum(task_loss) / len(task_loss),
                     "task": task_id + 1
                })

            if self.is_e_field and task_id !=0:
                self._append_loss_on_first_task()

            if task_id % self.save_model_interval == 0:
                self.generate_data_for_img_plots_recon(task_id)
            
            self.store_validate_encoded(sum(list_of_encoded) / len(list_of_encoded), task_id)
            self.validate_prev_tasks(task_id)
            self._save_model(self.run_name,self.model_path, task_id)

        if self.last_task_included:   
            self.generate_data_for_img_plots_recon(self.number_of_tasks - 1)
        self.plot_reconstructed_data_grid_task_wise()
        self.log_prev_tasks_data()
        
        self.validate_class_wise()
