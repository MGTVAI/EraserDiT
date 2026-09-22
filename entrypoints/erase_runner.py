"""CLI output path resolution."""

from pathlib import Path


def resolve_output_file_name(
    output_path: str,
    default_name: str = "eraserdit_output.mp4",
) -> tuple[str, str]:
    """Split user CLI output path into output dir + file name."""
    output = Path(output_path).expanduser().resolve()
    if output.suffix:
        return str(output.parent), output.name
    return str(output), default_name
