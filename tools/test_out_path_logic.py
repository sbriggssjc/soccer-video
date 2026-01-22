from __future__ import annotations

from pathlib import Path
import tempfile

from tools.render_follow_unified import _resolve_output_path, _temp_output_path


def test_out_path_logic() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        input_path = root / "input.mp4"

        out_dir = root / "renders"
        final_out, is_file = _resolve_output_path(str(out_dir), input_path, "FOLLOW")
        assert not is_file
        assert final_out.suffix == ".mp4"
        temp_out = _temp_output_path(final_out)
        assert temp_out.suffix == ".mp4"

        file_out = root / "custom_name.mp4"
        final_file_out, is_file = _resolve_output_path(str(file_out), input_path, "FOLLOW")
        assert is_file
        assert final_file_out == file_out


if __name__ == "__main__":
    test_out_path_logic()
