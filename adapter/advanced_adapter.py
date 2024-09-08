import torch
import torch.nn as nn
from .context_aware_generation import ContextAwareGeneration
from .memory_management import MemoryManager
from .semantic_understanding import SemanticUnderstanding
from language_modules.python_module import PythonModule
from language_modules.cpp_module import CppModule
from language_modules.javascript_module import JavaScriptModule
from language_modules.java_module import JavaModule
from language_modules.ruby_module import RubyModule
from language_modules.go_module import GoModule
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
        self.javascript_module = JavaScriptModule()
        self.java_module = JavaModule()
        self.ruby_module = RubyModule()
        self.go_module = GoModule()
        self.error_handler = ErrorHandler()
        self.code_style_controller = CodeStyleController()

        # Use CPU if CUDA is not available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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
        elif language == "javascript":
            code = self.javascript_module.generate_code(output)
        elif language == "java":
            code = self.java_module.generate_code(output)
        elif language == "ruby":
            code = self.ruby_module.generate_code(output)
        elif language == "go":
            code = self.go_module.generate_code(output)
        else:
            raise ValueError(f"Unsupported language: {language}")

        code = self.error_handler.check_and_fix(code, language)
        code = self.code_style_controller.apply_style(code, language)
        
        return code
