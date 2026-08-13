"""Execute and validate the statistical-analysis notebook."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient


root = Path(__file__).resolve().parents[1]
path = root / "notebooks" / "nbt_smallset_statistical_analysis.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(
    notebook,
    timeout=600,
    kernel_name="python3",
    resources={"metadata": {"path": str(root)}},
)
client.execute()
nbformat.write(notebook, path)
print(f"Executed {path}")
