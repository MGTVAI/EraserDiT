"""Origin-observable contracts shared by the LTX0.9.5 erase runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any
from typing import Mapping


SUPPORTED_LTX095_SP_DEGREES = frozenset({1, 2, 4})


def _optional_string(value: object) -> str | None:
    return None if value in (None, "", "N/A") else str(value)


@dataclass(frozen=True)
class VideoEncodingProfile:
    """Immutable origin-compatible FFmpeg output contract."""

    width: int
    height: int
    frame_rate: str
    encoder: str
    output_pix_fmt: str
    video_bitrate: str
    color_space: str | None
    color_transfer: str | None
    color_primaries: str | None
    color_range: str | None
    field_order: str

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> "VideoEncodingProfile":
        codec = metadata.get("codec_name")
        encoder_map = {"h264": "libx264", "prores": "prores_ks"}
        if not isinstance(codec, str) or codec.lower() not in encoder_map:
            raise ValueError(f"unsupported source codec: {codec!r}")

        frame_rate = metadata.get("fps_fraction")
        if not isinstance(frame_rate, str) or (
            "/" not in frame_rate and not frame_rate.isdecimal()
        ):
            raise ValueError(f"invalid rational frame rate: {frame_rate!r}")
        try:
            fraction = Fraction(frame_rate)
        except (ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"invalid rational frame rate: {frame_rate!r}") from exc
        if fraction <= 0:
            raise ValueError(f"invalid rational frame rate: {frame_rate!r}")

        raw_field_order = metadata.get("field_order")
        field_order_map = {
            None: "progressive",
            "progressive": "progressive",
            "top": "top_field_first",
            "top_field_first": "top_field_first",
            "tb": "top_field_first",
            "tt": "top_field_first",
            "tff": "top_field_first",
            "bottom": "bottom_field_first",
            "bottom_field_first": "bottom_field_first",
            "bb": "bottom_field_first",
            "bt": "bottom_field_first",
            "bff": "bottom_field_first",
            "interlaced": "interlaced",
        }
        if raw_field_order not in field_order_map:
            raise ValueError(f"unsupported field_order: {raw_field_order!r}")

        bitrate = metadata.get("bit_rate")
        video_bitrate = "10000000" if bitrate in (None, "", "N/A") else str(bitrate)
        return cls(
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            frame_rate=f"{fraction.numerator}/{fraction.denominator}",
            encoder=encoder_map[codec.lower()],
            output_pix_fmt=str(metadata.get("pix_fmt") or "yuv420p"),
            video_bitrate=video_bitrate,
            color_space=_optional_string(metadata.get("color_space")),
            color_transfer=_optional_string(metadata.get("color_transfer")),
            color_primaries=_optional_string(metadata.get("color_primaries")),
            color_range=_optional_string(metadata.get("color_range")),
            field_order=field_order_map[raw_field_order],
        )

    def ffmpeg_output_args(self) -> tuple[str, ...]:
        args = [
            "-vcodec",
            self.encoder,
            "-b:v",
            self.video_bitrate,
            "-pix_fmt",
            self.output_pix_fmt,
            "-r",
            self.frame_rate,
        ]
        for option, value in (
            ("-colorspace", self.color_space),
            ("-color_trc", self.color_transfer),
            ("-color_primaries", self.color_primaries),
            ("-color_range", self.color_range),
        ):
            if value is not None:
                args.extend((option, value))
        if self.field_order == "top_field_first":
            args.extend(("-flags", "+ildct+ilme", "-vf", "setfield=tff"))
        elif self.field_order == "bottom_field_first":
            args.extend(("-flags", "+ildct+ilme", "-vf", "setfield=bff"))
        elif self.field_order == "interlaced":
            args.extend(("-flags", "+ildct+ilme"))
        return tuple(args)


def resolve_spatial_alignment(sp_degree: int) -> tuple[int, int]:
    """Return origin-compatible ``(align_w, align_h)`` for an SP topology."""
    if isinstance(sp_degree, bool) or not isinstance(sp_degree, int):
        raise TypeError("sp_degree must be an integer")
    if sp_degree not in SUPPORTED_LTX095_SP_DEGREES:
        raise ValueError(f"unsupported LTX095 sp_degree: {sp_degree}")
    exponent = int(math.log2(sp_degree))
    align_w = 32 * (2 ** math.ceil(exponent / 2))
    align_h = 32 * (2 ** math.floor(exponent / 2))
    return int(align_w), int(align_h)


def resolve_runtime_sp_degree(server_args: Any) -> int:
    """Read the effective SP degree without introducing a second config source."""
    context = getattr(server_args, "parallel_context", None)
    if context is None or not bool(getattr(context, "enabled", False)):
        return 1
    plan = getattr(context, "plan", None)
    if plan is None:
        raise ValueError("enabled parallel context must provide an SP plan")
    return int(getattr(plan, "sp_degree"))
