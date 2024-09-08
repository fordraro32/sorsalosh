import torch
import torch.nn as nn
from .context_aware_generation import ContextAwareGeneration
from .memory_management import MemoryManager
from .multi_gpu_utils import distribute_model
from .semantic_understanding import SemanticUnderstanding
from language_modules.python_module import PythonModule
from language_modules.cpp_module import CppModule
from utils.error_handling import ErrorHandler
from utils.code_style_control import CodeStyleController

class AdvancedAdapterLayer(nn.Module):
    def __init__(self, config):
        super(AdvancedAdapterLayer, self).__init__()
        self.down_proj = nn.Linear(config.input_dim, config.bottleneck_dim)
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(config.bottleneck_dim, config.input_dim)
        self.use_memory = config.use_memory
        if self.use_memory:
            self.memory = nn.Parameter(torch.zeros(config.input_dim))

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        if self.use_memory:
            x = x + self.memory
        return x + residual

class AdvancedAdapter(nn.Module):
    def __init__(self, config):
        super(AdvancedAdapter, self).__init__()
        self.config = config
        self.adapter_layer = AdvancedAdapterLayer(config)
        self.context_aware_generation = ContextAwareGeneration(config)
        self.memory_manager = MemoryManager(config)
        self.semantic_understanding = SemanticUnderstanding()
        self.python_module = PythonModule()
        self.cpp_module = CppModule()
        self.error_handler = ErrorHandler()
        self.code_style_controller = CodeStyleController()

        # Distribute model across available GPUs
        self.model = distribute_model(self, config.num_gpus)

    def forward(self, x):
        x = self.adapter_layer(x)
        x = self.context_aware_generation(x)
        x = self.semantic_understanding(x)
        return x

    def generate(self, input_ids, attention_mask, language="python"):
        with torch.no_grad():
            output = self.forward(input_ids)
        
        if language == "python":
            code = self.python_module.generate_code(output)
        elif language == "cpp":
            code = self.cpp_module.generate_code(output)
        else:
            raise ValueError(f"Unsupported language: {language}")

        code = self.error_handler.check_and_fix(code, language)
        code = self.code_style_controller.apply_style(code, language)
        
        return code
