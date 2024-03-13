import torch
import torch.nn as nn
import numpy as np
import time
import random
from utilities import save_visual, save_visual_multi, filter_dims, validate_model
from plot_box_predictions import load_particles, plot_particles

def train_AE(model, optimizer,
             scheduler, epoch, valid_data_loader, wandb, directory,
             info_image_path="", property_ = "positions",
             log_visual_report_every_tb = 10,
             log_validation_loss_every_tb = 30
             ):
    
    config = wandb.config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    #Calculate and print the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    
    patience = 20
    slow_improvement_patience = 10
    no_improvement_count = 0
    slow_improvement_count = 0
    min_valid_loss = np.inf
    start_epoch = 0


    start_time = time.time()
    for i_epoch in range(start_epoch, config["num_epochs"]):   

        loss_overall = []
        for timeBatchIndex in range(len(epoch)):
            loss_avg = []
            timeBatch = epoch[timeBatchIndex]
            
            start_timebatch = time.time()
            for particleBatchIndex in range(len(timeBatch)):
                optimizer.zero_grad()
                
                phase_space, _ = timeBatch[particleBatchIndex]
                
                #TODO do this in the loader. Saves double code.
                phase_space = filter_dims[property_](phase_space)
                
                phase_space = phase_space.to(device)
                
                loss, _ = model(phase_space)
                
                loss_avg.append(loss.item())
                loss.backward()
                optimizer.step()
                
            end_timebatch = time.time()
            elapsed_timebatch = end_timebatch - start_timebatch
            
            loss_timebatch_avg = sum(loss_avg)/len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            timeInfo = f"e: {i_epoch}, tb:{timeBatchIndex}, "
            print(timeInfo +' last timebatch loss: {}, avg_loss: {}, time: {}'.format(loss.item(), 
                                                                                      loss_timebatch_avg, 
                                                                                      elapsed_timebatch),
            
                                                                          flush=True)
            wandb_log_dict={
                "Epoch": i_epoch,
                "tb":(i_epoch)*len(timeBatch) + timeBatchIndex,
                "loss_timebatch_avg_loss": loss_timebatch_avg
            }

            if timeBatchIndex%log_visual_report_every_tb==0 and property_ not in  ["all","momentum_force"]:
                save_visual(model, timeBatch, wandb, timeInfo, info_image_path, property_)
            elif timeBatchIndex%log_visual_report_every_tb==0:
                particles = load_particles()
                plot_particles(particles=particles, model=model, wandb_obj=wandb)
                #save_visual_multi(model, timeBatch, wandb, timeInfo, info_image_path, property_=property_)
            
            if timeBatchIndex%log_validation_loss_every_tb==0:
                # Perform validation
                val_loss_overall_avg = validate_model(model, valid_data_loader, 
                                                      property_, device)
                wandb_log_dict.update({
                "val_loss_overall_avg": val_loss_overall_avg})
            
            wandb.log(wandb_log_dict)
            
        loss_overall_avg = sum(loss_overall)/len(loss_overall)  
        
                    
        if min_valid_loss > val_loss_overall_avg:     
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{val_loss_overall_avg:.6f}) \t Saving The Model')
            min_valid_loss = val_loss_overall_avg
            no_improvement_count = 0
            slow_improvement_count = 0
            # Saving State Dict
            torch.save(model.state_dict(), 
                       directory + '/best_model_',
                       _use_new_zipfile_serialization=False)
        else:
            no_improvement_count += 1
            if loss_overall_avg - min_valid_loss <= 0.001:  # Adjust this threshold as needed
                slow_improvement_count += 1
        
        scheduler.step()
        
        # if no_improvement_count >= patience or slow_improvement_count >= slow_improvement_patience:
        #     break  # Stop training
        
    # Code or process to be measured goes here
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.6f} seconds")
