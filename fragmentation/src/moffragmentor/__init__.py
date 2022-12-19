# -*- coding: utf-8 -*-
"""MOFFragmentor aims to provide Python abstractions for reticular chemistry."""
__version__ = "0.0.1"
from loguru import logger

from .mof import MOF  # noqa: F401

logger.disable("moffragmentor")
