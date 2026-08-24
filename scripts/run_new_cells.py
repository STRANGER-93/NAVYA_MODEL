import copy
import importlib.util
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def load_module(p):
    spec = importlib.util.spec_from_file_location("cells_mod", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    nb_path, mod_path, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])
    mod = load_module(mod_path)
    nb = nbformat.read(nb_path, as_version=4)
    tmp = nbformat.v4.new_notebook()
    tmp.metadata = copy.deepcopy(nb.metadata)
    tmp.cells = [nbformat.v4.new_code_cell(mod.SETUP)]
    for src in mod.CELLS:
        tmp.cells.append(nbformat.v4.new_code_cell(src))
    client = NotebookClient(tmp, timeout=timeout, kernel_name="mc-cycle",
                            allow_errors=False)
    client.execute(cwd=str(Path(nb_path).parent))
    if mod.HEADER:
        nb.cells.append(nbformat.v4.new_markdown_cell(mod.HEADER))
    for src, executed in zip(mod.CELLS, tmp.cells[1:]):
        new_cell = nbformat.v4.new_code_cell(src)
        new_cell["execution_count"] = executed.get("execution_count")
        new_cell["outputs"] = executed.get("outputs", [])
        nb.cells.append(new_cell)
    nbformat.write(nb, nb_path)
    print(f"appended {len(mod.CELLS)} executed cells to {Path(nb_path).name}")


if __name__ == "__main__":
    main()
