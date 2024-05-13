import random
from random import randint, sample, seed
seed(32)

class EpisodicMemory():
    '''
    Class representing an episodic memory buffer for continual learning.

    Args:
        total_mem_size (int): Total memory size for all tasks.
        total_tasks (int): Total number of tasks.
        gradient_ref_data_size (int, optional): Size of reference data for gradient-based methods (default: 0).

    Attributes:
        total_mem_size (int): Total memory size for all tasks.
        total_tasks (int): Total number of tasks.
        gradient_ref_data_size (int): Size of reference data for gradient-based methods.
        mem_size_per_task_list (list): List of memory sizes per task.
        copy_of_mem_size_per_task_list (list): Copy of memory sizes per task.
        ep_mem_list (list): List containing episodic memory data.
        _current_task_ids (list): List of current task IDs.

    Methods:
        _init_mem_per_task_list(): Initializes memory size per task.
        add_data_for_task(task_id, data, labels, pooling_index_list=None): Adds data for a specific task to the episodic memory.
        add_data_e_field(task_id, data, labels): Adds data for a specific task using a queue-like strategy when memory is full.
        get_data_for_reference_gradient(task_id): Retrieves data for reference gradient-based methods.
    '''
    def __init__(self, total_mem_size, total_tasks, gradient_ref_data_size=0):
        '''
        Initializes an EpisodicMemory instance.

        Args:
            total_mem_size (int): Total memory size for all tasks.
            total_tasks (int): Total number of tasks.
            gradient_ref_data_size (int, optional): Size of reference data for gradient-based methods (default: 0).
        '''
        self.total_mem_size = total_mem_size
        self.total_tasks = total_tasks
        self.gradient_ref_data_size = gradient_ref_data_size
        self.mem_size_per_task_list = self._init_mem_per_task_list()
        self.copy_of_mem_size_per_task_list = self.mem_size_per_task_list.copy()
        self.ep_mem_list = []
        self._current_task_ids = list(range(len(self.mem_size_per_task_list)))
    
    def _init_mem_per_task_list(self):
        '''
        Initializes memory size per task.

        Returns:
            list: List of memory sizes per task.
        '''
        mem_task, rem_mem_task = divmod(self.total_mem_size, self.total_tasks - 1)
        mem_size_per_task_list = []
        for _ in range(self.total_tasks - 1):
            if rem_mem_task != 0:
                mem_size_per_task_list.append(mem_task + 1)
                rem_mem_task -= 1
            else:
                mem_size_per_task_list.append(mem_task)
        
        return [i for i in mem_size_per_task_list if i != 0]

    def add_data_for_task(self, task_id, data, labels, pooling_index_list=None):
        '''
        Adds data for a specific task to the episodic memory.

        Args:
            task_id (int): ID of the task.
            data (list): List of data samples.
            labels (list): List of corresponding labels.
            pooling_index_list (list, optional): List of pooling indices (default: None).
        '''
        if self.mem_size_per_task_list[task_id] != 0:
            data_len = len(data)
            if data_len <= self.mem_size_per_task_list[task_id]:
                for d,l in zip(data,labels): self.ep_mem_list.append((d.cpu(),l.cpu()))
                self.mem_size_per_task_list[task_id] = self.mem_size_per_task_list[task_id] - data_len
            else:
                for d,l in zip(data[:self.mem_size_per_task_list[task_id]],labels[:self.mem_size_per_task_list[task_id]]): self.ep_mem_list.append((d.cpu(),l.cpu()))
                self.mem_size_per_task_list[task_id] = 0
        else:
            pass

    def add_data_e_field(self, task_id, data, labels):
        '''
        Adds data for a specific task using a queue-like strategy when memory is full.

        Args:
            task_id (int): ID of the task.
            data (list): List of data samples.
            labels (list): List of corresponding labels.
        '''
        if task_id not in self._current_task_ids:
            self.ep_mem_list.pop(0)
            self._current_task_ids.pop(0)
            self._current_task_ids.append(task_id)
            for d,l in zip(data[:1],labels[:1]): self.ep_mem_list.append((d.cpu(),l.cpu()))
            self.mem_size_per_task_list[-1] = self.mem_size_per_task_list[-1] - 1
        else:
            idx = self._current_task_ids.index(task_id)
            if self.mem_size_per_task_list[idx] != 0:
                data_len = len(data)
                if data_len <= self.mem_size_per_task_list[idx]:
                        for d,l in zip(data,labels): self.ep_mem_list.append((d.cpu(),l.cpu()))
                        self.mem_size_per_task_list[idx] = self.mem_size_per_task_list[idx] - data_len
                else:
                    for d,l in zip(data[:self.mem_size_per_task_list[idx]],labels[:self.mem_size_per_task_list[idx]]): self.ep_mem_list.append((d.cpu(),l.cpu()))
                    self.mem_size_per_task_list[idx] = 0
    
    def get_data_for_reference_gradient(self, task_id):
        '''
        Retrieves data for reference gradient-based methods.

        Args:
            task_id (int): ID of the task.

        Returns:
            list: List of data samples and labels.
        '''
        if self.gradient_ref_data_size == 0:
            chosen_int = randint(0,task_id - 1)
            print(chosen_int, "random choice for sgem")
            idx = self._current_task_ids.index(chosen_int)
            cnt = 0
            for i in range(idx):
                cnt += self.copy_of_mem_size_per_task_list[i]
            return self.ep_mem_list[cnt: self.copy_of_mem_size_per_task_list[idx]]
        else:
            if task_id == self.total_tasks - 1:
                return sample(self.ep_mem_list, self.gradient_ref_data_size)

            idx = self._current_task_ids.index(task_id)
            cnt = 0
            for i in range(idx):
                cnt += self.copy_of_mem_size_per_task_list[i]
            if cnt < self.gradient_ref_data_size:
                return self.ep_mem_list[:cnt]
            return sample(self.ep_mem_list[:cnt], self.gradient_ref_data_size)
