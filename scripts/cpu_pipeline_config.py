"""CPU pipeline settings owned by the S4 experiment scripts."""
import os

CPU_PIPELINE_DEFAULTS = {
    "MGERASE_VIDEO_UINT8_GPU_NORMALIZE": "0",
    "MGERASE_VIDEO_UINT8_GPU_CLEAR_CACHE": "0",
    "MGERASE_PIPE_READ_THREADS": "1",
    "MGERASE_MASK_PIPE_R_ONLY": "0",
    "MGERASE_STREAMING_COPY_OPT": "0",
    "MGERASE_MASK_WINDOW_BUFFER_OPT": "0",
    "MGERASE_OUTPUT_UINT8_GPU_CONVERT": "0",
    "MGERASE_WRITER_HANDOFF_OPT": "0",
    "MGERASE_MASK_THRESHOLD_INPLACE": "0",
    "MGERASE_STYLE_UINT8_GPU_UPLOAD": "0",
    "MGERASE_MASK_DILATE_CAST_OPT": "0",
    "MGERASE_WINDOW_UINT8_VALIDATION_OPT": "0",
    "MGERASE_FIRST_WINDOW_BORROW": "0",
    "MGERASE_TEXT_SESSION_CACHE": "0",
    "MGERASE_MASK_SCAN_REUSE": "0",
}
S4_ENV = {name: "1" for name in CPU_PIPELINE_DEFAULTS}
S4_ENV["MGERASE_PIPE_READ_THREADS"] = "2"


def benchmark_environment():
    """Record selected values; do not enable optimizations implicitly."""
    return {name: os.environ.get(name, default)
            for name, default in CPU_PIPELINE_DEFAULTS.items()}
