# StreamedML
Framework for training machine learning models from streamed data 

Extend your model for CL methods
Inherit ContinualLearner class

```python
class MLP(ContinualLearner):
```

For EWC:
```python
# after learning a task - Estimate Fisher
model.estimate_fisher(current_task_data_set, loss_func, is_mlp = True)

# while training get EWC Loss
ewc_loss = regularizer_strength * model.ewc_loss()
```

For Replay-Based Methods:
```python
# while training
reference_data = sampled_Data_from_replay_memory()
if layerwise:
    # calculating reference gradients
    model.calculate_ref_gradients_layerwise(reference_data)

    # optimization step
    model.overwrite_grad_layerwise()

# A-GEM Case
else:
    # calculating reference gradients
    model.calculate_ref_gradients(reference_data)

    # optimization step
    model.overwrite_grad()

# After Successful Task Training - append data to Replay Memory
# Examples are in ReplayTrainer.py
```

Episodic Memory implementation - utils/EpisodicMemory.py