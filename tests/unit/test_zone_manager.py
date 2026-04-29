"""
tests/unit/test_zone_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ZoneManager polygon logic.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.utils.zone_manager import Zone, ZoneManager


def make_square_zone(camera_id: str = "cam-01", zone_id: str = "z1") -> Zone:
    return Zone(
        id=zone_id,
        name="Test Zone",
        camera_id=camera_id,
        polygon=np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32),
    )


class TestZoneManager:
    def test_add_and_retrieve_zone(self):
        zm = ZoneManager()
        zone = make_square_zone()
        zm.add_zone(zone)
        assert len(zm.zones_for_camera("cam-01")) == 1

    def test_zones_filtered_by_camera(self):
        zm = ZoneManager()
        zm.add_zone(make_square_zone("cam-01", "z1"))
        zm.add_zone(make_square_zone("cam-02", "z2"))
        assert len(zm.zones_for_camera("cam-01")) == 1
        assert zm.zones_for_camera("cam-01")[0].id == "z1"

    def test_remove_zone(self):
        zm = ZoneManager()
        zm.add_zone(make_square_zone())
        removed = zm.remove_zone("z1")
        assert removed is True
        assert zm.zones_for_camera("cam-01") == []

    def test_remove_nonexistent_zone(self):
        zm = ZoneManager()
        assert zm.remove_zone("does-not-exist") is False

    def test_inactive_zone_not_checked(self):
        zm = ZoneManager()
        zone = make_square_zone()
        zm.add_zone(zone)
        zm.set_active("z1", False)
        # centre of zone — would normally trigger
        result = zm.check_intrusion("cam-01", [40, 40, 60, 60])
        assert result == []

    def test_intrusion_detected_inside_zone(self):
        zm = ZoneManager()
        zm.add_zone(make_square_zone())
        result = zm.check_intrusion("cam-01", [40, 40, 60, 60])
        assert "z1" in result

    def test_no_intrusion_outside_zone(self):
        zm = ZoneManager()
        zm.add_zone(make_square_zone())
        # box entirely outside the 0-100 square
        result = zm.check_intrusion("cam-01", [200, 200, 300, 300])
        assert result == []

    def test_multiple_zones_multiple_triggers(self):
        zm = ZoneManager()
        zm.add_zone(make_square_zone("cam-01", "z1"))
        zone2 = Zone(
            id="z2",
            name="Zone 2",
            camera_id="cam-01",
            polygon=np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.int32),
        )
        zm.add_zone(zone2)
        # box overlaps both zones
        result = zm.check_intrusion("cam-01", [60, 60, 80, 80])
        assert "z1" in result
        assert "z2" in result

    def test_point_in_polygon_centre(self):
        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert ZoneManager._point_in_polygon((5, 5), polygon) is True

    def test_point_outside_polygon(self):
        polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        assert ZoneManager._point_in_polygon((15, 15), polygon) is False
