from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_typescript as ts_typescript

from exctractor import find_source_files

TS_LANGUAGE = Language(ts_typescript.language_typescript())

parser = Parser(TS_LANGUAGE)


source_path = Path("sample_repository/src/api.ts")
source_bytes = source_path.read_bytes()

tree = parser.parse(source_bytes)

# print(tree.root_node)

# root = tree.root_node

# for child in root.named_children:
#     print(child.type, child.start_point, child.end_point)

#     declaration = child.child_by_field_name("declaration")

#     if declaration is not None:
#         print("  declaration:", declaration.type)

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
    return source_bytes[
        node.start_byte:node.end_byte
    ].decode("utf-8")

def extract_functions(path):
    source_bytes = Path(path).read_bytes()
    tree = parser.parse(source_bytes)

    functions = []

    # 1. Найти function_declaration
    function_nodes = find_nodes_by_type(
        tree.root_node,
        "function_declaration"
    )

    # 2. Добавить каждую в functions
    for node in function_nodes:
        name_node = node.child_by_field_name("name")

        functions.append({
            "path": str(path),
            "name": get_node_text(name_node, source_bytes),
            "kind": "function_declaration",
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "code": get_node_text(node, source_bytes)
        })

    # 3. Найти variable_declarator с arrow_function
    variable_nodes = find_nodes_by_type(
        tree.root_node,
        "variable_declarator"
    )

    # 4. Добавить каждую в functions
    for node in variable_nodes:
        name_node = node.child_by_field_name("name")
        value_node = node.child_by_field_name("value")

        if value_node is None:
            continue

        if value_node.type != "arrow_function":
            continue

        functions.append({
            "path": str(path),
            "name": get_node_text(name_node, source_bytes),
            "kind": "arrow_function",
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "code": get_node_text(node, source_bytes)
        })

    # 5. Отсортировать по start_line
    functions.sort(
        key=lambda function: function["start_line"]
    )

    # 6. Вернуть functions
    return functions

def extract_repository_functions(repository_path):
    all_functions = []

    paths = find_source_files(repository_path)

    for path in paths:
        functions = extract_functions(path)
        all_functions.extend(functions)

    return all_functions

# functions = extract_repository_functions(
#     "sample_repository"
# )

# for function in functions:
#     print(
#         f'{function["path"]}:'
#         f'{function["start_line"]}-'
#         f'{function["end_line"]} '
#         f'{function["name"]} '
#         f'({function["kind"]})'
#     )