"""Contract tests that run without downloading PyTorch or model weights."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import unittest

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/moda_fashionsiglip_multiview"


def load_image_view_functions():
    source = (PACKAGE / "retriever.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"square_pad", "center_square"}
    ]
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = {"Image": Image}
    exec(compile(module, "retriever.py", "exec"), namespace)
    return namespace["square_pad"], namespace["center_square"]


class PublicContractTest(unittest.TestCase):
    def test_metadata_and_frozen_result_agree(self):
        config = json.loads((ROOT / "config.json").read_text())
        recipe = json.loads((PACKAGE / "recipe.json").read_text())
        summary = json.loads(
            (ROOT / "benchmark_results/summary.json").read_text()
        )

        self.assertEqual(
            config["base_model_parameters"],
            recipe["base_model_parameters"],
        )
        self.assertEqual(
            recipe["base_model_parameters"],
            summary["candidate"]["neural_parameters"],
        )
        self.assertEqual(recipe["candidate_id"], "late-maxview-b010")
        self.assertEqual(summary["significant_wins"], 4)
        self.assertEqual(summary["significant_losses"], 0)
        self.assertEqual(summary["point_wins"], 6)

    def test_fusion_weights_are_a_convex_combination(self):
        recipe = json.loads((PACKAGE / "recipe.json").read_text())
        parent_weight = recipe["late_fusion_parent_weight"]
        max_view_weight = recipe["late_fusion_max_view_weight"]
        self.assertTrue(math.isclose(parent_weight, 0.9))
        self.assertTrue(math.isclose(max_view_weight, 0.1))
        self.assertTrue(math.isclose(parent_weight + max_view_weight, 1.0))

    def test_actual_image_view_functions_have_locked_geometry(self):
        square_pad, center_square = load_image_view_functions()
        image = Image.new("RGB", (6, 2), (255, 0, 0))

        padded = square_pad(image)
        cropped = center_square(image)

        self.assertEqual(padded.size, (6, 6))
        self.assertEqual(padded.getpixel((0, 0)), (128, 128, 128))
        self.assertEqual(padded.getpixel((0, 2)), (255, 0, 0))
        self.assertEqual(cropped.size, (2, 2))
        self.assertEqual(cropped.getpixel((0, 0)), (255, 0, 0))

    def test_no_upstream_weights_are_bundled(self):
        forbidden_suffixes = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
        bundled = [
            path
            for path in ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in forbidden_suffixes
        ]
        self.assertEqual(bundled, [])


if __name__ == "__main__":
    unittest.main()
