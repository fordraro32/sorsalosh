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
import ast
import re

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
        
        refactored_code, suggestions = self.refactor_and_optimize(code, language)
        
        return refactored_code, suggestions

    def refactor_and_optimize(self, code, language):
        suggestions = []
        
        if language == "python":
            refactored_code, new_suggestions = self._refactor_python(code)
        elif language == "cpp":
            refactored_code, new_suggestions = self._refactor_cpp(code)
        elif language == "javascript":
            refactored_code, new_suggestions = self._refactor_javascript(code)
        elif language == "java":
            refactored_code, new_suggestions = self._refactor_java(code)
        elif language == "ruby":
            refactored_code, new_suggestions = self._refactor_ruby(code)
        elif language == "go":
            refactored_code, new_suggestions = self._refactor_go(code)
        else:
            raise ValueError(f"Unsupported language for refactoring: {language}")
        
        suggestions.extend(new_suggestions)
        
        return refactored_code, suggestions

    def _refactor_python(self, code):
        suggestions = []
        tree = ast.parse(code)
        
        # Check for unused imports
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    imports.add(n.name)
            elif isinstance(node, ast.ImportFrom):
                for n in node.names:
                    imports.add(f"{node.module}.{n.name}")
        
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
        
        unused_imports = imports - used_names
        if unused_imports:
            suggestions.append(f"Consider removing unused imports: {', '.join(unused_imports)}")
        
        # Check for long functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and len(node.body) > 20:
                suggestions.append(f"Consider refactoring the function '{node.name}' as it's quite long ({len(node.body)} lines)")
        
        # Suggest list comprehensions for simple loops
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and len(node.body) == 1:
                if isinstance(node.body[0], ast.Assign) and isinstance(node.body[0].value, ast.Call):
                    suggestions.append(f"Consider using a list comprehension instead of the for loop at line {node.lineno}")
        
        return code, suggestions

    def _refactor_cpp(self, code):
        suggestions = []
        
        # Check for using namespace std
        if "using namespace std;" in code:
            suggestions.append("Consider avoiding 'using namespace std;' and instead use explicit 'std::' prefixes")
        
        # Check for long functions (crude approximation)
        functions = re.findall(r'(\w+\s+\w+\s*\([^)]*\)\s*\{[^}]*\})', code)
        for func in functions:
            if func.count('\n') > 30:
                suggestions.append(f"Consider refactoring the function starting with '{func[:50]}...' as it's quite long")
        
        # Suggest const correctness
        if re.search(r'\b(?:int|float|double|char)\s+\w+\s*\([^)]*\)\s*const\b', code) is None:
            suggestions.append("Consider using 'const' for member functions that don't modify object state")
        
        return code, suggestions

    def _refactor_javascript(self, code):
        suggestions = []
        
        # Check for var usage
        if 'var ' in code:
            suggestions.append("Consider using 'let' or 'const' instead of 'var' for better scoping")
        
        # Check for long functions (crude approximation)
        functions = re.findall(r'function\s+\w+\s*\([^)]*\)\s*\{[^}]*\}', code)
        for func in functions:
            if func.count('\n') > 30:
                suggestions.append(f"Consider refactoring the function starting with '{func[:50]}...' as it's quite long")
        
        # Suggest arrow functions for simple functions
        if re.search(r'function\s*\(\w*\)\s*{\s*return', code):
            suggestions.append("Consider using arrow functions for simple one-line functions")
        
        return code, suggestions

    def _refactor_java(self, code):
        suggestions = []
        
        # Check for long methods (crude approximation)
        methods = re.findall(r'(\w+\s+\w+\s*\([^)]*\)\s*\{[^}]*\})', code)
        for method in methods:
            if method.count('\n') > 30:
                suggestions.append(f"Consider refactoring the method starting with '{method[:50]}...' as it's quite long")
        
        # Suggest using final for parameters
        if not re.search(r'\b(?:public|private|protected)\s+\w+\s+\w+\s*\((?:[^,)]*\bfinal\b[^,)]*,?)*[^)]*\)', code):
            suggestions.append("Consider using 'final' for method parameters that are not reassigned")
        
        # Suggest using StringBuilder for string concatenation in loops
        if re.search(r'for\s*\([^)]+\)\s*\{[^}]*\+=[^}]*\}', code):
            suggestions.append("Consider using StringBuilder for string concatenation in loops")
        
        return code, suggestions

    def _refactor_ruby(self, code):
        suggestions = []
        
        # Check for long methods (crude approximation)
        methods = re.findall(r'def\s+\w+(?:\([^)]*\))?\s*\n[^end]*end', code)
        for method in methods:
            if method.count('\n') > 20:
                suggestions.append(f"Consider refactoring the method starting with '{method[:50]}...' as it's quite long")
        
        # Suggest using symbols instead of strings as hash keys
        if re.search(r'{\s*[\'"][^\'"]+[\'"]\s*=>', code):
            suggestions.append("Consider using symbols instead of strings as hash keys")
        
        # Suggest using unless instead of if !
        if 'if !' in code:
            suggestions.append("Consider using 'unless' instead of 'if !' for negative conditions")
        
        return code, suggestions

    def _refactor_go(self, code):
        suggestions = []
        
        # Check for long functions (crude approximation)
        functions = re.findall(r'func\s+\w+\([^)]*\)\s*(?:\w+\s*)?\{[^}]*\}', code)
        for func in functions:
            if func.count('\n') > 30:
                suggestions.append(f"Consider refactoring the function starting with '{func[:50]}...' as it's quite long")
        
        # Suggest using := for short variable declarations
        if re.search(r'\bvar\s+\w+\s*=', code):
            suggestions.append("Consider using ':=' for short variable declarations instead of 'var'")
        
        # Suggest using range-based for loops
        if re.search(r'for\s+\w+\s*:=\s*0;\s*\w+\s*<\s*len\([^)]+\);\s*\w+\+\+', code):
            suggestions.append("Consider using range-based for loops (for i := range slice) instead of C-style for loops")
        
        return code, suggestions
