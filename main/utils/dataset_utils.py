import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms

from StreamDataReader.StreamBuffer import StreamBuffer
from utils.cifar100_coarse import CIFAR100Coarse


class UnknownDatasetError(ValueError):
    def __init__(self, msg=None):
        if msg is None:
            msg = 'unknown dataset name'
        super().__init__(msg)


class SubDataset(Dataset):
    '''
    Sub-samples a dataset, taking only those samples with label in [sub_labels].
    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.
    '''

    def __init__(self, original_dataset, sub_labels, target_transform=None):
        '''
        Initializes the SubDataset.

        Parameters:
            original_dataset (Dataset): The original dataset to be sub-sampled.
            sub_labels (list): The list of labels to include in the sub-sample.
            target_transform (callable, optional): A function/transform to apply to the target label (default: None).
        '''
        super().__init__()
        self.dataset = original_dataset
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform

    def __len__(self):
        '''
        Returns the length of the sub-dataset.

        Returns:
            int: The length of the sub-dataset.
        '''
        return len(self.sub_indeces)

    def __getitem__(self, index):
        '''
        Retrieves an item from the sub-dataset.

        Parameters:
            index (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the sample and its transformed target label.
        '''
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
        return sample

class EfieldDataset(Dataset):
    '''
    Dataset class for electric field data.

    Attributes:
        data_dir (str): The directory containing the data files.
        iterations (list): List of iteration IDs for tasks.
        train_dim (str, optional): The dimension for training data (default: None).
    '''

    def __init__(self, data_dir, num_tasks, dim=None):
        '''
        Initializes the EfieldDataset.

        Parameters:
            data_dir (str): The directory containing the data files.
            num_tasks (int): Number of tasks.
            dim (str, optional): The dimension for training data (default: None).
        '''
        self.data_dir = data_dir
        self.iterations = num_tasks
        self.train_dim = dim
    
    def __len__(self):
        '''
        Returns the length of the dataset.

        Returns:
            int: The number of tasks.
        '''
        return len(self.iterations)
    
    def __getitem__(self, idx):
        '''
        Retrieves an item from the dataset.

        Parameters:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: A tuple containing the normalized tensor data and the iteration ID.
        '''
        iteration_id = self.iterations[idx]
        data = np.load(os.path.join(
            self.data_dir, "data_{}.npy".format(iteration_id * 100)))
        if self.train_dim is not None:
            idx = DIMENSION_MAP[self.train_dim]
            data = data[idx]
            dim = (1,128,1280,128)
        dim = (3,128,1280,128)

        norm_data = []
        #mean = [ -1.40394059e-06,  1.57518510e-04, -1.12175585e-05]
        #std = [ 0.07121728, 0.00651213, 0.07119355]
        mean = [-5.55989313e-06,  1.71693718e-04, -9.08668555e-06]
        std = [0.07360622, 0.00663636, 0.07351927]
        if dim[0] == 1:
            norm_data.append((data[idx] - mean[idx])/std[idx])
        else:
            for i in range(3):
                norm_data.append((data[i] - mean[i])/std[i])
        norm_tensor = torch.from_numpy(np.array(norm_data))

        return norm_tensor.view(-1,dim[1],dim[2],dim[3]) , iteration_id

def _permutate_image_pixels(image, permutation):
    '''
    Permutes the pixels of an image tensor.

    Parameters:
        image (torch.Tensor): The input image tensor.
        permutation (torch.Tensor): The permutation tensor.

    Returns:
        torch.Tensor: Permuted image tensor.
    '''
    if permutation is None:
        return image

    c, h, w = image.size()
    image = image.view(-1, c)
    image = image[permutation, :]
    
    return image

DIMENSION_MAP = {
    'x':0,
    'y':1,
    'z':2
}

AVAILABLE_TRANSFORMS = {
    'mnist': [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ],
    'cifar10' : [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ],
    'cifar100' : [
        transforms.ToTensor(),
        transforms.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276))
    ]
}    


def _get_dataset(
        name,
        datasets_dir,
        train=True,
        download=True,
        permutation=None,
):
    '''
    Retrieves a dataset based on its name.

    Parameters:
        name (str): Name of the dataset.
        datasets_dir (str): Directory containing the datasets.
        train (bool): Whether to retrieve the training set (default: True).
        download (bool): Whether to download the dataset if not found (default: True).
        permutation (torch.Tensor): Permutation tensor (default: None).

    Returns:
        Dataset: The requested dataset.
    '''
    if 'mnist' in name:
        dataset_class = datasets.MNIST
        dataset_name = 'mnist'
        if name == 'p-mnist':
            dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[dataset_name],
            transforms.Lambda(lambda x: _permutate_image_pixels(x, permutation)),
            ])
        else:
            dataset_transform = transforms.Compose([
            *AVAILABLE_TRANSFORMS[dataset_name],])
    elif 'cifar10' in name:
        dataset_class = datasets.CIFAR10
        dataset_name = 'cifar10'
        if train:
            dataset_transform =  transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *AVAILABLE_TRANSFORMS[dataset_name],])
        else:
            dataset_transform =  transforms.Compose([
                *AVAILABLE_TRANSFORMS[dataset_name],])
    
    elif 'cifar-100' in name:
        dataset_class = CIFAR100Coarse
        dataset_name = 'cifar100'
        if train:
            dataset_transform =  transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                *AVAILABLE_TRANSFORMS[dataset_name],])
        else:
            dataset_transform =  transforms.Compose([
                *AVAILABLE_TRANSFORMS[dataset_name],])

    else:
        raise UnknownDatasetError()

    return dataset_class(
        os.path.join(datasets_dir, '{name}'.format(name=dataset_name)),
        train=train, download=download, transform=dataset_transform)


def get_tasks_datasets(
        name,
        datasets_dir,
        classes,
        num_tasks,
        train=True,
        permutations=None):
    '''
    Retrieves datasets for multiple tasks.

    Parameters:
        name (str): Name of the dataset.
        datasets_dir (str): Directory containing the datasets.
        classes (int): Number of classes.
        num_tasks (int): Number of tasks.
        train (bool): Whether to retrieve the training set (default: True).
        permutations (list): List of permutation tensors (default: None).

    Returns:
        list: List of datasets for each task.
    '''
    out_datasets = []
    if permutations:
        for perm in permutations:
            out_datasets.append(_get_dataset(
                name,
                datasets_dir,
                train=train,
                download=True,
                permutation=perm,
            ))
        return out_datasets
    og_dataset = _get_dataset(name, datasets_dir, train, download=True)
    classes_per_task = int(np.floor(classes / num_tasks))
    labels_per_task = [
                list(np.array(range(classes_per_task)) + classes_per_task * task_id) for task_id in range(num_tasks)
            ]
    for labels in labels_per_task:
        out_datasets.append(SubDataset(og_dataset,labels))

    return out_datasets
