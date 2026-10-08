import ast
import json
import unittest

from benchmark.run import ROOT


class NotebookTests(unittest.TestCase):
    def test_generated_notebook_has_no_outputs_and_all_python_compiles(self):
        notebook = json.loads((ROOT / "notebooks/retrieval_rerank_colab.ipynb").read_text(encoding="utf-8"))
        ids = []
        for index, cell in enumerate(notebook["cells"]):
            ids.append(cell["id"])
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")
        self.assertEqual(len(ids), len(set(ids)))

    def test_probe_string_compiles_as_a_subprocess_program(self):
        notebook = json.loads((ROOT / "notebooks/retrieval_rerank_colab.ipynb").read_text(encoding="utf-8"))
        probes = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                for node in ast.walk(ast.parse("".join(cell["source"]))):
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "probe" for t in node.targets):
                        probes.append(ast.literal_eval(node.value))
        self.assertEqual(len(probes), 1)
        compile(probes[0], "gpu-probe", "exec")


if __name__ == "__main__":
    unittest.main()
