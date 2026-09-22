"""Local entrypoints for the minimal EraserDiT runtime."""

from utils.logging_utils import globally_suppress_loggers

globally_suppress_loggers(
    "transformers.modeling_utils",
    "diffusers.configuration_utils",
)
