"""
visp.utils.zone_manager
~~~~~~~~~~~~~~~~~~~~~~~~
Polygon-based restricted zone management.

Zones are defined as lists of (x, y) vertices in pixel coordinates
relative to the native camera resolution. The manager can check
whether any bounding box from a detection result overlaps a zone.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    id: str
    name: str
    camera_id: str
    polygon: np.ndarray          # shape (N, 2), dtype int
    active: bool = True
    color: str = "#FF4444"       # for dashboard rendering

    @classmethod
    def from_dict(cls, data: dict) -> "Zone":
        return cls(
            id=data["id"],
            name=data["name"],
            camera_id=data["camera_id"],
            polygon=np.array(data["polygon"], dtype=np.int32),
            active=data.get("active", True),
            color=data.get("color", "#FF4444"),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "camera_id": self.camera_id,
            "polygon": self.polygon.tolist(),
            "active": self.active,
            "color": self.color,
        }


class ZoneManager:
    """
    Manages restricted zones per camera and checks bounding box intersections.

    Example
    -------
    >>> zm = ZoneManager()
    >>> zm.add_zone(Zone(id="z1", name="Server Room", camera_id="cam-01",
    ...                  polygon=np.array([[0,0],[100,0],[100,100],[0,100]])))
    >>> zm.check_intrusion("cam-01", bounding_box=[40, 40, 60, 60])
    ['z1']
    """

    def __init__(self) -> None:
        self._zones: dict[str, Zone] = {}  # keyed by zone.id

    # ── Mutation ──────────────────────────────────────────────────

    def add_zone(self, zone: Zone) -> None:
        self._zones[zone.id] = zone
        logger.info("Zone added: id=%s name=%r camera=%s", zone.id, zone.name, zone.camera_id)

    def remove_zone(self, zone_id: str) -> bool:
        if zone_id in self._zones:
            del self._zones[zone_id]
            logger.info("Zone removed: id=%s", zone_id)
            return True
        return False

    def set_active(self, zone_id: str, active: bool) -> None:
        if zone_id in self._zones:
            self._zones[zone_id].active = active

    # ── Queries ───────────────────────────────────────────────────

    def zones_for_camera(self, camera_id: str) -> list[Zone]:
        return [z for z in self._zones.values() if z.camera_id == camera_id and z.active]

    def check_intrusion(
        self,
        camera_id: str,
        bounding_box: Sequence[int],
    ) -> list[str]:
        """
        Return IDs of all active zones that the bounding box overlaps.

        Parameters
        ----------
        bounding_box: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bounding_box
        # Sample the four corners and centre of the box
        test_points = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2),
            ((x1 + x2) // 2, (y1 + y2) // 2),
        ]

        triggered: list[str] = []
        for zone in self.zones_for_camera(camera_id):
            if any(self._point_in_polygon(pt, zone.polygon) for pt in test_points):
                triggered.append(zone.id)

        return triggered

    # ── Persistence ───────────────────────────────────────────────

    def load_from_file(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        for item in data:
            self.add_zone(Zone.from_dict(item))
        logger.info("Loaded %d zones from %s", len(data), path)

    def save_to_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps([z.to_dict() for z in self._zones.values()], indent=2))

    # ── Internal ──────────────────────────────────────────────────

    @staticmethod
    def _point_in_polygon(point: tuple[int, int], polygon: np.ndarray) -> bool:
        """Ray-casting algorithm for point-in-polygon test."""
        x, y = point
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside
