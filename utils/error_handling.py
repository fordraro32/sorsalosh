import re

class ErrorHandler:
    def __init__(self):
        self.common_errors = {
            'python': {
                'IndentationError': re.compile(r'IndentationError'),
                'NameError': re.compile(r'NameError'),
                'SyntaxError': re.compile(r'SyntaxError'),
            },
            'cpp': {
                'UndeclaredVariable': re.compile(r"'(\w+)' was not declared in this scope"),
                'MissingSemicolon': re.compile(r"expected ';' before"),
                'MissingInclude': re.compile(r"'(\w+)' file not found"),
            }
        }

    def check_and_fix(self, code, language):
        if language == 'python':
            return self._fix_python_errors(code)
        elif language == 'cpp':
            return self._fix_cpp_errors(code)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _fix_python_errors(self, code):
        lines = code.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            # Fix indentation errors
            if re.match(r'^\s*\S', line):
                fixed_lines.append('    ' + line)
            else:
                fixed_lines.append(line)

            # Add missing colons
            if any(keyword in line for keyword in ['if', 'else', 'for', 'while', 'def', 'class']) and not line.strip().endswith(':'):
                fixed_lines[-1] += ':'

        return '\n'.join(fixed_lines)

    def _fix_cpp_errors(self, code):
        # Add missing semicolons
        code = re.sub(r'([^;{])\s*$', r'\1;', code, flags=re.MULTILINE)

        # Add missing includes
        if 'cout' in code and '#include <iostream>' not in code:
            code = '#include <iostream>\n' + code

        return code

    def add_debugging_code(self, code, language):
        if language == 'python':
            return self._add_python_debugging(code)
        elif language == 'cpp':
            return self._add_cpp_debugging(code)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _add_python_debugging(self, code):
        debug_code = "import traceback\n\ntry:\n"
        indented_code = '\n'.join('    ' + line for line in code.split('\n'))
        debug_code += indented_code
        debug_code += "\nexcept Exception as e:\n    print(f'Error: {e}')\n    traceback.print_exc()"
        return debug_code

    def _add_cpp_debugging(self, code):
        debug_code = "#include <iostream>\n#include <stdexcept>\n\nint main() {\n    try {\n"
        indented_code = '\n'.join('        ' + line for line in code.split('\n'))
        debug_code += indented_code
        debug_code += "\n    } catch (const std::exception& e) {\n        std::cerr << \"Error: \" << e.what() << std::endl;\n    }\n    return 0;\n}"
        return debug_code
