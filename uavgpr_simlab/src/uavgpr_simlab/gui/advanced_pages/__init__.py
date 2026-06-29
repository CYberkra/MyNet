"""Page builders for the advanced engineering GUI."""

from uavgpr_simlab.gui.advanced_pages.dashboard_tab import AdvancedDashboardTabWidgets, build_advanced_dashboard_tab
from uavgpr_simlab.gui.advanced_pages.env_tab import AdvancedEnvTabWidgets, build_advanced_env_tab
from uavgpr_simlab.gui.advanced_pages.generation_tab import AdvancedGenerationTabWidgets, build_advanced_generation_tab
from uavgpr_simlab.gui.advanced_pages.history_tab import AdvancedHistoryTabWidgets, build_advanced_history_tab
from uavgpr_simlab.gui.advanced_pages.model_preview_tab import AdvancedModelPreviewTabWidgets, build_advanced_model_preview_tab
from uavgpr_simlab.gui.advanced_pages.preflight_tab import AdvancedPreflightTabWidgets, build_advanced_preflight_tab
from uavgpr_simlab.gui.advanced_pages.qc_tab import AdvancedQcTabWidgets, build_advanced_qc_tab
from uavgpr_simlab.gui.advanced_pages.queue_tab import AdvancedQueueTabWidgets, build_advanced_queue_tab
from uavgpr_simlab.gui.advanced_pages.real_csv_tab import AdvancedRealCsvTabWidgets, build_advanced_real_csv_tab
from uavgpr_simlab.gui.advanced_pages.train_tab import AdvancedTrainTabWidgets, build_advanced_train_tab

__all__ = [
    "AdvancedDashboardTabWidgets",
    "build_advanced_dashboard_tab",
    "AdvancedEnvTabWidgets",
    "build_advanced_env_tab",
    "AdvancedGenerationTabWidgets",
    "build_advanced_generation_tab",
    "AdvancedHistoryTabWidgets",
    "build_advanced_history_tab",
    "AdvancedModelPreviewTabWidgets",
    "build_advanced_model_preview_tab",
    "AdvancedPreflightTabWidgets",
    "build_advanced_preflight_tab",
    "AdvancedQcTabWidgets",
    "build_advanced_qc_tab",
    "AdvancedQueueTabWidgets",
    "build_advanced_queue_tab",
    "AdvancedRealCsvTabWidgets",
    "build_advanced_real_csv_tab",
    "AdvancedTrainTabWidgets",
    "build_advanced_train_tab",
]
