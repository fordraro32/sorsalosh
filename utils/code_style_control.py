import re

class CodeStyleController:
    def __init__(self):
        self.python_style = {
            'indentation': 4,
            'max_line_length': 79,
            'snake_case': True,
        }
        self.cpp_style = {
            'indentation': 2,
            'max_line_length': 80,
            'camel_case': True,
        }

    def apply_style(self, code, language):
        if language == 'python':
            return self._apply_python_style(code)
        elif language == 'cpp':
            return self._apply_cpp_style(code)
        else:
            raise ValueError(f"Unsupported language: {language}")

    def _apply_python_style(self, code):
        lines = code.split('\n')
        styled_lines = []
        for line in lines:
            # Apply indentation
            stripped_line = line.lstrip()
            indentation = len(line) - len(stripped_line)
            styled_line = ' ' * (indentation // self.python_style['indentation'] * self.python_style['indentation']) + stripped_line

            # Enforce max line length
            if len(styled_line) > self.python_style['max_line_length']:
                # Simple line break, in practice you'd want more sophisticated line breaking
                styled_line = styled_line[:self.python_style['max_line_length']] + ' \\'

            # Convert to snake_case if enabled
            if self.python_style['snake_case']:
                styled_line = re.sub(r'(?<!^)(?=[A-Z])', '_', styled_line).lower()

            styled_lines.append(styled_line)

        return '\n'.join(styled_lines)

    def _apply_cpp_style(self, code):
        lines = code.split('\n')
        styled_lines = []
        for line in lines:
            # Apply indentation
            stripped_line = line.lstrip()
            indentation = len(line) - len(stripped_line)
            styled_line = ' ' * (indentation // self.cpp_style['indentation'] * self.cpp_style['indentation']) + stripped_line

            # Enforce max line length
            if len(styled_line) > self.cpp_style['max_line_length']:
                # Simple line break, in practice you'd want more sophisticated line breaking
                styled_line = styled_line[:self.cpp_style['max_line_length']] + ' \\'

            # Convert to camelCase if enabled
            if self.cpp_style['camel_case']:
                words = styled_line.split('_')
                styled_line = words[0] + ''.join(word.capitalize() for word in words[1:])

            styled_lines.append(styled_line)

        return '\n'.join(styled_lines)

    def set_style(self, language, **kwargs):
        if language == 'python':
            self.python_style.update(kwargs)
        elif language == 'cpp':
            self.cpp_style.update(kwargs)
        else:
            raise ValueError(f"Unsupported language: {language}")
