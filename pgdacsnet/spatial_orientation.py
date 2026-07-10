"""Spatial orientation contracts for the YingShan UAV-GPR survey.

Canonical arrays always remain in acquisition/CSV trace order.  Engineering
profiles and several report figures use a separate left-to-right display order.
This module keeps those two concepts explicit so training data are never
silently reversed merely to match a figure.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LineOrientation:
    line: str
    engineering_profile: str
    profile_left: str
    profile_right: str
    profile_display_flip: bool
    confidence: str
    evidence: str


# Evidence was cross-checked against the plan/profile PDFs, the UavGPR report
# figures, and trace-ordered GNSS endpoints.  L1 and X1 lack a standalone
# engineering profile in the supplied profile archive, so their display
# contracts are deliberately marked medium confidence.
ORIENTATION_REGISTRY: dict[str, LineOrientation] = {
    "Line3": LineOrientation(
        line="Line3", engineering_profile="3-3′",
        profile_left="3 / ZK07 / south", profile_right="3′ / ZK08 / north",
        profile_display_flip=True, confidence="high",
        evidence="Engineering profile is south-to-north; CSV acquisition runs north/ZK08 to south/ZK07.",
    ),
    "Line6": LineOrientation(
        line="Line6", engineering_profile="6-6′",
        profile_left="6 / ZK09 / south", profile_right="6′ / ZK10 / north",
        profile_display_flip=True, confidence="high",
        evidence="Engineering profile and migrated terrain descend south-to-north; CSV acquisition runs north to south.",
    ),
    "Line7": LineOrientation(
        line="Line7", engineering_profile="7-7′",
        profile_left="7 / west / ZK09 side", profile_right="7′ / east / Line3 crossing side",
        profile_display_flip=False, confidence="high",
        evidence="Engineering profile and CSV acquisition both run west-to-east.",
    ),
    "Line9": LineOrientation(
        line="Line9", engineering_profile="9-9′",
        profile_left="9 / west", profile_right="9′ / east / ZK08 side",
        profile_display_flip=True, confidence="high",
        evidence="Engineering profile runs west-to-east; CSV acquisition runs east/ZK08 to west.",
    ),
    "LineL1": LineOrientation(
        line="LineL1", engineering_profile="report-only L1 section",
        profile_left="east / Line3 crossing side", profile_right="west / Line6 crossing side",
        profile_display_flip=False, confidence="medium",
        evidence="No standalone engineering-profile PDF supplied; main migrated section agrees with CSV east-to-west order despite conflicting aerial arrows.",
    ),
    "LineX1": LineOrientation(
        line="LineX1", engineering_profile="report-only X1 section",
        profile_left="south / L1 crossing side", profile_right="north / Line9 crossing side",
        profile_display_flip=True, confidence="medium",
        evidence="No standalone engineering-profile PDF supplied; report section places L1 crossing in the reversed CSV display order.",
    ),
}


def get_line_orientation(line: str) -> LineOrientation:
    try:
        return ORIENTATION_REGISTRY[str(line)]
    except KeyError as exc:
        raise KeyError(f"No spatial orientation contract registered for {line!r}") from exc


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Initial geodesic bearing from the first trace to the final trace."""
    lon1r, lat1r, lon2r, lat2r = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2r - lon1r
    x = math.sin(dlon) * math.cos(lat2r)
    y = math.cos(lat1r) * math.sin(lat2r) - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def compass_from_bearing(bearing: float) -> str:
    labels = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return labels[int((float(bearing) + 22.5) // 45) % 8]


def orientation_metadata(line: str, longitude: np.ndarray, latitude: np.ndarray) -> dict[str, Any]:
    longitude = np.asarray(longitude, dtype=np.float64).reshape(-1)
    latitude = np.asarray(latitude, dtype=np.float64).reshape(-1)
    if longitude.size < 2 or longitude.size != latitude.size:
        raise ValueError("longitude/latitude must be matching vectors with at least two traces")
    contract = get_line_orientation(line)
    bearing = bearing_deg(longitude[0], latitude[0], longitude[-1], latitude[-1])
    return {
        **asdict(contract),
        "acquisition_bearing_deg": float(bearing),
        "acquisition_compass": compass_from_bearing(bearing),
        "trace0_longitude": float(longitude[0]),
        "trace0_latitude": float(latitude[0]),
        "last_longitude": float(longitude[-1]),
        "last_latitude": float(latitude[-1]),
    }


def profile_index_order(width: int, line: str) -> np.ndarray:
    order = np.arange(int(width), dtype=np.int64)
    return order[::-1] if get_line_orientation(line).profile_display_flip else order


def align_array_for_display(values: np.ndarray, line: str, axis: int = -1, *, orientation: str = "profile") -> np.ndarray:
    """Return a display view while preserving acquisition-order source arrays."""
    arr = np.asarray(values)
    if orientation not in {"acquisition", "profile"}:
        raise ValueError("orientation must be 'acquisition' or 'profile'")
    if orientation == "profile" and get_line_orientation(line).profile_display_flip:
        return np.flip(arr, axis=axis)
    return arr


def display_distance_axis(
    acquisition_distance_m: np.ndarray,
    line: str,
    *,
    orientation: str = "profile",
) -> np.ndarray:
    """Create a left-to-right distance axis starting at zero for a display view."""
    distance = np.asarray(acquisition_distance_m, dtype=np.float64).reshape(-1)
    if distance.size == 0:
        return distance
    if np.any(np.diff(distance) < -1e-9):
        raise ValueError("acquisition distance must be non-decreasing")
    if orientation == "acquisition":
        return distance - distance[0]
    if orientation != "profile":
        raise ValueError("orientation must be 'acquisition' or 'profile'")
    if get_line_orientation(line).profile_display_flip:
        return distance[-1] - distance[::-1]
    return distance - distance[0]
