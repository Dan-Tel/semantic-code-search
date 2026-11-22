from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_typescript as ts_typescript

from .source_files import find_source_files


TS_LANGUAGE = Language(ts_typescript.language_typescript())

parser = Parser(TS_LANGUAGE)


def find_nodes_by_type(root, target_type):
    result = []
    stack = [root]

    while stack:
        node = stack.pop()

        if node.type == target_type:
            result.append(node)

        stack.extend(reversed(node.named_children))

    return result


def get_node_text(node, source_bytes):
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")


def _function_record(path, node, kind, source_bytes):
    return {
        "path": str(path),
        "name": get_node_text(node.child_by_field_name("name"), source_bytes),
        "kind": kind,
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "code": get_node_text(node, source_bytes),
    }


def extract_functions(path):
    source_bytes = Path(path).read_bytes()
    tree = parser.parse(source_bytes)
    declarations = find_nodes_by_type(tree.root_node, "function_declaration")
    functions = [
        _function_record(path, node, "function_declaration", source_bytes)
        for node in declarations
    ]

    for node in find_nodes_by_type(tree.root_node, "variable_declarator"):
        value = node.child_by_field_name("value")
        if value is not None and value.type == "arrow_function":
            functions.append(_function_record(path, node, "arrow_function", source_bytes))

    return sorted(functions, key=lambda function: function["start_line"])


def extract_repository_functions(repository_path):
    all_functions = []

    paths = find_source_files(repository_path)

    for path in paths:
        functions = extract_functions(path)
        all_functions.extend(functions)

    return all_functions
