"""Page builders for the UavGPR-SimLab easy GUI."""

from uavgpr_simlab.gui.pages.batch_page import BatchPageWidgets, build_batch_page
from uavgpr_simlab.gui.pages.history_page import HistoryPageWidgets, build_history_page
from uavgpr_simlab.gui.pages.home_page import HomePageWidgets, build_home_page
from uavgpr_simlab.gui.pages.model_preview_page import ModelPreviewPageWidgets, build_model_preview_page
from uavgpr_simlab.gui.pages.project_page import ProjectPageWidgets, build_project_page
from uavgpr_simlab.gui.pages.settings_page import SettingsPageWidgets, build_settings_help_page

__all__ = [
    "BatchPageWidgets",
    "HistoryPageWidgets",
    "HomePageWidgets",
    "ModelPreviewPageWidgets",
    "ProjectPageWidgets",
    "SettingsPageWidgets",
    "build_batch_page",
    "build_history_page",
    "build_home_page",
    "build_model_preview_page",
    "build_project_page",
    "build_settings_help_page",
]
