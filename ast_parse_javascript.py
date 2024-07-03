from slimit.parser import Parser
from slimit.visitors import nodevisitor
from slimit import ast

def find_js_methods(source_code):
    parser = Parser()
    tree = parser.parse(source_code)
    classes = []

    def get_node_text(node):
        return source_code[node.start:end_node.end]

    def get_comments(node):
        comments = []
        current = node
        while current:
            if hasattr(current, 'leadingComments'):
                comments.extend([comment.value for comment in current.leadingComments])
            current = getattr(current, 'parent', None)
        return comments

    for node in nodevisitor.visit(tree):
        if isinstance(node, ast.ClassDeclaration):
            class_name = node.identifier.value
            methods = []

            for element in node.body:
                if isinstance(element, ast.MethodDefinition):
                    method_name = element.function.identifier.value
                    method_implementation = get_node_text(element.function.body)
                    comments = get_comments(element)
                    methods.append({
                        "method_name": method_name,
                        "method_implementation": method_implementation,
                        "comments": comments
                    })

            classes.append({"class_name": class_name, "methods": methods})

    return classes

# Example usage
js_code = """
// This is a sample class
class MyClass {
    /**
     * This is method one
     */
    methodOne() {
        console.log("Hello");
    }

    // This is method two
    methodTwo() {
        // Some code
    }
}
"""
find_js_methods(js_code)
