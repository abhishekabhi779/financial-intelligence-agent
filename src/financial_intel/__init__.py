"""
Channel Intelligence Agent — Core Package
"""

__version__ = "0.1.0"
__author__ = "Abhishek"

from financial_intel.config import get_settings, load_settings_from_yaml, get_llm_config

__all__ = [
    "get_settings",
    "load_settings_from_yaml",
    "get_llm_config",
]