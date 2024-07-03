import ast
import javalang
import subprocess
import json

def find_python_methods(source_code):
    tree = ast.parse(source_code)
    classes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            methods = [
                {
                    "method_name": n.name,
                    "method_implementation": ast.unparse(n) if hasattr(ast, 'unparse') else "Not available in this Python version"
                }
                for n in node.body if isinstance(n, ast.FunctionDef)
            ]
            classes.append({"class_name": class_name, "methods": methods})

    return classes

def find_java_methods(source_code):
    tree = javalang.parse.parse(source_code)
    lines = source_code.splitlines()
    classes = []

    # Extract comments
    def extract_comments(source_code):
        comments = []
        in_comment = False
        comment = ''
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('//'):
                comments.append({'line': i + 1, 'comment': stripped})
            elif stripped.startswith('/*'):
                in_comment = True
                comment = stripped
                if stripped.endswith('*/'):
                    in_comment = False
                    comments.append({'line': i + 1, 'comment': comment})
                    comment = ''
            elif in_comment:
                comment += ' ' + stripped
                if stripped.endswith('*/'):
                    in_comment = False
                    comments.append({'line': i + 1, 'comment': comment})
                    comment = ''
        return comments

    comments = extract_comments(source_code)

    def get_code_fragment(start_pos, end_pos):
        if start_pos[0] == end_pos[0]:
            return lines[start_pos[0] - 1][start_pos[1] - 1:end_pos[1] - 1]
        else:
            fragment = lines[start_pos[0] - 1][start_pos[1] - 1:]
            for i in range(start_pos[0], end_pos[0] - 1):
                fragment += "\n" + lines[i]
            fragment += "\n" + lines[end_pos[0] - 1][:end_pos[1] - 1]
            return fragment

    for path, node in tree.filter(javalang.tree.ClassDeclaration):
        class_name = node.name
        methods = []
        for method in node.methods:
            start_pos = method.position
            if method.body:
                last_statement = method.body[-1]
                end_pos = (last_statement.position[0], len(lines[last_statement.position[0] - 1]))
            else:
                end_pos = (method.position[0], method.position[1] + len(method.name))
            method_implementation = get_code_fragment(start_pos, end_pos)

            # Find associated comments
            associated_comments = [comment['comment'] for comment in comments if comment['line'] < start_pos[0]]

            methods.append({
                "method_name": method.name,
                "method_implementation": method_implementation
            })
        classes.append({"class_name": class_name, "methods": methods})

    return classes


def find_js_ts_methods(file_path):
    result = subprocess.run(['node', 'esprima_parse.js', file_path], capture_output=True, text=True)
    classes = json.loads(result.stdout)
    return classes

def find_methods(file_path, language):
    with open(file_path, 'r') as file:
        source_code = file.read()
    
    if language == 'python':
        return find_python_methods(source_code)
    elif language == 'java':
        return find_java_methods(source_code)
    elif language in ['javascript', 'typescript']:
        with open('temp_code.js', 'w') as temp_file:
            temp_file.write(source_code)
        return find_js_ts_methods('temp_code.js')
    else:
        raise ValueError("Unsupported language")

# Example usage
# print("Python Classes and Methods:", find_methods('scan_code_base/example.py', 'python'))
print("Java Classes and Methods:", find_methods('scan_code_base/java/DefaultWebClient.java', 'java'))
# print("JavaScript Classes and Methods:", find_methods('scan_code_base/example.js', 'javascript'))
