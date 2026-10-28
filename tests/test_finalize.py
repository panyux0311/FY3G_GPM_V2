from __future__ import annotations

import json
from pathlib import Path
import tarfile
import tempfile
import unittest
import numpy as np

from schema import GRID_SHAPE
from utils.reporting import create_result_archive, select_single_input_track, track_crop


class TestFinalize(unittest.TestCase):
    def test_track_crop(self) -> None:
        valid = np.zeros((3, 2, 10, *GRID_SHAPE), dtype=bool)
        valid[:, :, :, 520:540, 800:820] = True
        rows, columns = track_crop(valid, padding=4)
        self.assertEqual((rows.start, rows.stop), (516, 544))
        self.assertEqual((columns.start, columns.stop), (796, 824))

    def test_single_track_selector(self) -> None:
        valid = np.zeros((3, 2, 10, *GRID_SHAPE), dtype=bool)
        valid[0, :, :, 10:12, 10:16] = True  # 12 cells
        valid[1, :, :, 30:33, 30:33] = True  # 9 cells
        selected, info = select_single_input_track(valid)
        self.assertEqual(info["selected_input_hour_index"], 0)
        self.assertEqual(info["selected_input_hour_before_forecast"], 3)
        self.assertEqual(info["selected_track_cells"], 12)
        self.assertTrue(selected[10:12, 10:16].all())
        self.assertFalse(selected[30:33, 30:33].any())

    def test_create_result_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir) / "project"
            output = project / "outputs/gpm"
            output.mkdir(parents=True)
            config = project / "configs/gpm_surface_v1.yaml"
            config.parent.mkdir(parents=True)
            config.write_text("model: {}\n", encoding="utf-8")
            (output / "eval.json").write_text(json.dumps({"csi": 0.5}), encoding="utf-8")
            checkpoint = output / "checkpoints/best.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"mock_checkpoint")

            archive = project / "result.tar.gz"
            res = create_result_archive(project, output, config, archive)
            self.assertFalse(res["checkpoint_included"])
            self.assertTrue(Path(res["sha256_file"]).exists())

            with tarfile.open(archive, "r:gz") as tar:
                names = tar.getnames()
            self.assertTrue(any("eval.json" in name for name in names))
            self.assertFalse(any(".pt" in name for name in names))
