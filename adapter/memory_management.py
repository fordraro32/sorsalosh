import torch

class MemoryManager:
    def __init__(self, config):
        self.max_memory = config.max_memory
        self.current_memory = 0

    def allocate(self, tensor):
        tensor_size = tensor.element_size() * tensor.nelement()
        if self.current_memory + tensor_size > self.max_memory:
            raise MemoryError("Not enough memory to allocate tensor")
        self.current_memory += tensor_size
        return tensor.to('cuda')

    def free(self, tensor):
        tensor_size = tensor.element_size() * tensor.nelement()
        self.current_memory -= tensor_size
        del tensor
        torch.cuda.empty_cache()

    def check_memory(self):
        return f"Current memory usage: {self.current_memory / 1024 / 1024 / 1024:.2f} GB"
