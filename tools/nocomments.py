import pathlib
import subprocess
import sys

from tree_sitter_language_pack import get_parser

LANGS = {".rs": "rust", ".py": "python", ".cu": "cpp", ".cuh": "cpp"}

hits = []
for path in sys.argv[1:]:
    lang = LANGS.get(pathlib.Path(path).suffix)
    if not lang:
        continue
    src = subprocess.run(["git", "show", f":{path}"], capture_output=True).stdout
    stack = [get_parser(lang).parse(src).root_node]
    while stack:
        node = stack.pop()
        if "comment" not in node.type:
            stack.extend(node.children)
            continue
        line = node.start_point[0] + 1
        text = node.text.decode(errors="replace").splitlines()[0]
        if line > 1 or not text.startswith("#!"):
            hits.append(f"  {path}:{line}: {text}")

if hits:
    print(f"BLOCKED: {len(hits)} comment(s) staged.\n")
    print("\n".join(hits))
    print("\nThis repo carries no comments. Make the code say it instead:")
    print("rename the thing, split the expression, or delete the dead branch.")
sys.exit(1 if hits else 0)
