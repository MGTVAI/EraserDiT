"""Local entrypoints for the minimal MGErase runtime."""

from utils.logging_utils import globally_suppress_loggers

globally_suppress_loggers(
    "transformers.modeling_utils",
    "diffusers.configuration_utils",
)
