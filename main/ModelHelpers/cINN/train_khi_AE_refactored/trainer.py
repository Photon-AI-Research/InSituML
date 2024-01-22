import torch
import torch.nn as nn
import numpy as np
import time 

def save_checkpoint(model,
                    optimizer,
                    path, 
                    last_loss,
                    min_valid_loss,
                    epoch,
                    wandb_run_id):
        
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'last_loss': last_loss.item(),
            'epoch': epoch,
            'min_valid_loss': min_valid_loss,
            'wandb_run_id': wandb_run_id,
        }

        torch.save(state, path + '/model_' + str(epoch))


def train_AE(model, criterion, optimizer, scheduler, epoch, wandb):
    
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
        for tb in range(len(epoch)):
            loss_avg = []
            timebatch = epoch[tb]
            
            batch_idx = 0
            start_timebatch = time.time()
            for b in range(len(timebatch)):
                batch_idx += 1
                optimizer.zero_grad()
                phase_space, _ = timebatch[b]
                
                phase_space = phase_space.permute(0, 2, 1).to(device)
                
                output = model(phase_space)
                
                loss = criterion(output.transpose(2,1).contiguous(),
                                 phase_space.transpose(2,1).contiguous())

                loss_avg.append(loss.item())
                loss.backward()
                optimizer.step()
                
            end_timebatch = time.time()
            elapsed_timebatch = end_timebatch - start_timebatch
            
            loss_timebatch_avg = sum(loss_avg)/len(loss_avg)
            loss_overall.append(loss_timebatch_avg)
            print('i_epoch:{}, tb: {}, last timebatch loss: {}, avg_loss: {}, time: {}'.format(i_epoch,tb,loss.item(), loss_timebatch_avg, elapsed_timebatch), flush=True)
    
        loss_overall_avg = sum(loss_overall)/len(loss_overall)  
    
        if min_valid_loss > loss_overall_avg:     
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{loss_overall_avg:.6f}) \t Saving The Model')
            min_valid_loss = loss_overall_avg
            no_improvement_count = 0
            slow_improvement_count = 0
            # Saving State Dict
            torch.save(model.state_dict(), directory + '/best_model_', _use_new_zipfile_serialization=False)
        else:
            no_improvement_count += 1
            if loss_overall_avg - min_valid_loss <= 0.001:  # Adjust this threshold as needed
                slow_improvement_count += 1
        
        scheduler.step()
        
        # Log the loss and accuracy values at the end of each epoch
        wandb.log({
            "Epoch": i_epoch,
            "loss_timebatch_avg_loss": loss_timebatch_avg,
            "loss_overall_avg": loss_overall_avg,
            "min_valid_loss": min_valid_loss,
        })
            
        
        # if no_improvement_count >= patience or slow_improvement_count >= slow_improvement_patience:
        #     break  # Stop training
        
    # Code or process to be measured goes here
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Total elapsed time: {elapsed_time:.6f} seconds")
