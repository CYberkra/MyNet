"""Controller mixins for the product-oriented easy GUI.

These mixins keep page orchestration out of ``easy_window.py`` while still
using the same QWidget instances created by page builders.
"""

from .home_project_controller import HomeProjectControllerMixin
from .model_preview_controller import ModelPreviewControllerMixin
from .batch_controller import BatchControllerMixin
from .history_controller import HistoryControllerMixin
from .settings_controller import SettingsControllerMixin

__all__ = [
    "HomeProjectControllerMixin",
    "ModelPreviewControllerMixin",
    "BatchControllerMixin",
    "HistoryControllerMixin",
    "SettingsControllerMixin",
]
