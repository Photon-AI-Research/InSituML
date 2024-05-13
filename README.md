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

## Time-dependent toy data generation and training

* `src/insituml/toy_data/generate.py`
* `examples/streaming_toy_data/{data_vis.py,simple_train.py}`

## Install

All source code is located in `src/insituml`. See also `pyproject.toml`.

```sh
$ git clone ...
$ cd /path/to/InSituML
```

Then one of these. We recommend a dev (a.k.a. "editable") install
(`pip install -e .`).

```sh
# Dev install to default location, e.g. in a venv
#   ~/.virtualenvs/awesome_venv/lib/python3.11/site-packages/insituml-0.0.0.dist-info/
$ python3 -m venv awesome_venv && . ./awesome_venv/bin/activate
awesome_venv $ pip install -e .

# Dev install in $HOME
$ pip install --user -e .

# Same, don't install dependencies
$ pip install --user --no-deps -e .
```

Low-tech alternative without an external tool:

```sh
$ export PYTHONPATH=/path/to/InSituML/src:$PYTHONPATH
```

Then

```py
from insituml.foo.bar import baz
```

## Tests

Run tests, skip tests marked with "examples" that may generate interactive
plots.

```sh
$ pytest
```

Also run examples. The "examples" marker is defined in `conftest.py`.

```sh
$ pytest --examples
```

## Description of old code

### main/ModelHelpers

#### main/ModelHelpers/DeviceHelper.py

The module contains functions:
    -  Get type of the default device: cpu or cuda 
    -  Move tensor(s) to chosen device 

DeviceDataLoader(): wrap a dataloader such that each batch is not only indexed but also instantly transfered to
the correct device

#### main/ModelHelpers/ContinualLearner.py

1. ContinualLearner(nn.Module): Class for realization of a neural network to be trained using CL (further Continual Learning).
There is saved some information on method to use (EWC by default), implementation of EWC related
methods:
    -  Fisher Value estimate 
    -  EWC Loss  
    -  Calculates refrence gradients based on sampled memory from previous episodes 
    -  Overwrite gradients for parameter update based on dot products as mentioned in A-GEM method 
    -  Layerswise gradient calculation and overwriting
2. EpisodicMemoryDataset(Dataset): Class to interface torch-dataset for episodic memory approach to train the model,
contains regular dataset methods: init, length and indexing

#### main/ModelHelpers/mlp.py

MLP(ContinualLearner): MLP architecture for an autoencoder to be trained in CL approach, contain methods:
-  xavier weigth initialization for conv2d, transposedConv2D, linear layers
-  create linear layers 
-  Encoder and Decoder Initialization: sequence of {nn.Linear, activation} blocks 
-  Inverse from AE: first decode then encode 
-  Save Checkpoint with all meta data (model's hyperparameters, CL training method)

#### main/ModelHelpers/Autoencoder2D.py

AutoEncoder2D(ContinualLearner): Autoencoder for 2D tensors (processing of images), inherited from ContinualLearning class
Contains methods:
-  xavier weigth initialization for conv2d, transposedConv2D, linear layers 
-  create layers of conv2d, transposedConv2D, linear architectures 
-  find an "n-th" half: value/2^n 
-  find flatten size of a linear layer 
-  Encoder Initialization: sequence of {Conv2D, activation, MaxPool of 2} blocks, nn.Flatten, linear layer, activation 
-  Decoder Initialization: linear layer, sequence of {TransposedConv2D, activation, nn.MaxUnpool2d(2)} blocks 
-  Inverse from AE: first decode then encode, uses Upsample instead of Unpool 
-  Save Checkpoint with all meta data (model's hyperparameters, CL training method) 


#### main/ModelHelpers/Autoencoder3D.py

AutoEncoder3D(ContinualLearner): Autoencoder for 3D tensors (processing of 3D volumes, e.g. distribution of a field in 3D space), inherited from ContinualLearning class
Contains methods:
-  xavier weigth initialization for conv3d, transposedConv3D, linear layers 
-  create layers of conv3d, transposedConv3D, linear architectures 
-  find an "n-th" half: value/2^n 
-  find flatten size of a linear layer 
-  Encoder Initialization: sequence of {Conv3D, activation, MaxPool of 2} blocks, nn.Flatten, linear layer, activation 
-  Decoder Initialization: linear layer, sequence of {TransposedConv2D, activation, nn.MaxUnpool2d(2)} blocks 
-  Inverse from AE: first decode then encode, uses Upsample instead of Unpool 
-  Save Checkpoint with all meta data (model's hyperparameters, CL training method) 
-  Split model between 2 GPUs: encoder is transfered to cuda:0, decoder to cuda:1 

#### main/ModelHelpers/MeshDimensionDataset.py
MeshDimensionDataset(Dataset): a wrap for dataset, used in main/ModelEvaluator.py, main/ModelTrainerTaskWise.py
There is additional method to save some chunks of data with some meta information, data is returned in a flatten form
within this class.

#### main/ModelHelpers/PlotHelper.py

Contains one method to create figures for wandb logging: groundtruth and prediction in each figure,
in "jet" and in "grey" scaling. The method takes min/max values for normalization of images, but normalization is
commented in the code.

## /main/utils

-- cifar_coarse.py
Dependencies on other files within the repository: none

Has one class:

- class CIFAR100Coarse:  Groups the original CFAR100 fine-grained classes into 20 coarse-grained classes. Inherits from CIFAR100 class.

-- dataset_utils.py
Dependencies on other files within the repository:
    - from StreamDataReader.StreamBuffer import StreamBuffer
    - from utils.cifar100_coarse import CIFAR100Coarse
    
Has two classes:
- class SubDataset: Sub-samples a dataset, taking only those samples with label in [sub_labels]. After this selection of samples has been made, it is possible to transform the target-labels, which can be useful when doing continual learning with fixed number of output units.
- class EFieldDataset: Handles dataset class for electric field data.


--  dist_utils.py
Dependencies on other files within the repository: None

A series of fucntions that provide essential functionality for setting up and managing distributed training processes in PyTorch, including process synchronization, distributed data parallelism, and distributed data sampling.

-- EpisodicMemory.py
Dependencies on other files within the repository: None

Has one class:
- class EpisodicMemory: Represents an episodic memory buffer for continual learning. It allows storing and retrieving data for different tasks, with options for handling memory overflow and providing reference data for gradient-based methods.

-- plot_helper.py
Dependencies on other files within the repository: None

Contains utility functions for plotting data: 
- "plot_reconstructed_data" plots original and reconstructed images side by side
- "plot_reconstructed_data_taskwise" plots original images and reconstructed images task-wise
- "plot_heatmap_df" plots a heatmap from DataFrame data


 

