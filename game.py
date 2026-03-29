import base64
from dataclasses import dataclass
from math import isnan, nan, asin, degrees
from random import uniform, choice
from typing import Any, Optional

import requests
from pyproj import Geod

from util import log_event, GOOGLE_MAPS_API_KEY

""" (lat, lon) """
LOCATION_DATABASE: list[tuple[float, float]] = [
    (35.6595, 139.7005),
    (-22.9711, -43.1822),
    (-33.9628, 18.4098),
    (64.1466, -21.9426),
    (1.2975414, 103.7779669)
]

_WGS84_GEOD = Geod(ellps="WGS84")


def wrap_degrees(degrees: float) -> float:
    while degrees < 0:
        degrees += 360
    while degrees >= 360:
        degrees -= 360
    return degrees


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def has_streetview(lat: float, lon: float) -> bool:
    url: str = (f"https://maps.googleapis.com/maps/api/streetview/metadata"
                f"?location={lat},{lon}"
                f"&key={GOOGLE_MAPS_API_KEY}")
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        return False
    data: dict[str, Any] = resp.json()
    return data.get("status", '') == "OK"


def find_nearby_streetview(lat: float, lon: float, radius_m: float) -> tuple[float, float]:
    url: str = (f"https://maps.googleapis.com/maps/api/streetview/metadata"
                f"?location={lat},{lon}"
                f"&radius={radius_m}"
                f"&key={GOOGLE_MAPS_API_KEY}")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()
    location: Optional[dict[str, Any]] = data.get("location")
    if location is None:
        return nan, nan
    return location["lat"], location["lng"]


@dataclass
class StreetViewImage:
    mime: str
    data: bytes


class Game:
    def __init__(self):
        self._tar_lon: float = nan
        self._tar_lat: float = nan
        self._cur_lon: float = nan  # degree
        self._cur_lat: float = nan  # degree
        self._pitch: float = nan  # degree
        self.heading: float = nan  # degree, clockwise from true north
        self.api_key: str = GOOGLE_MAPS_API_KEY

    def reset(self, lat: float, lon: float, heading: float = 0.0) -> None:
        self._cur_lat = self._tar_lat = lat
        self._cur_lon = self._tar_lon = lon
        self._pitch = 0.0
        self.heading = heading

    def set_to_random_street_view(self) -> None:
        for _ in range(50):
            lat: float = degrees(asin(uniform(-1.0, 1.0)))
            lon: float = uniform(-180.0, 180.0)
            streetview_latlon: tuple[float, float] = find_nearby_streetview(lat, lon, 200_000)
            if not any(map(isnan, streetview_latlon)):
                self.reset(*streetview_latlon)
                log_event(f"StreetView randomly chosen as {streetview_latlon}")
                return
        log_event(f"All 50 tries failed, fall back to database")
        self.reset(*choice(LOCATION_DATABASE))

    def _street_view_url(self, size: str = "1920x1280", fov: float = 100) -> str:
        if len(self.api_key) <= 0:
            raise ValueError("Google Maps API key is required")
        return (
            "https://maps.googleapis.com/maps/api/streetview"
            f"?size={size}"
            f"&location={self._cur_lat},{self._cur_lon}"
            f"&heading={self.heading}"
            f"&pitch={self._pitch}"
            f"&fov={fov}"
            f"&key={self.api_key}"
        )

    def render_image(self, size: str = "1920x1280", timeout: int = 20) -> str:
        """ Fetch the image of current Street View"""
        url: str = self._street_view_url(size=size)
        log_event(f"Fetching {url}")
        res = requests.get(url, timeout=timeout)
        res.raise_for_status()
        content_type = (res.headers.get("content-type") or "").split(";")[0].strip().lower()
        if not content_type.startswith("image/"):
            raise ValueError(f"Street View API returned non-image content: {content_type}")
        return base64.b64encode(res.content).decode("utf-8")

    def turn(self, delta_yaw: float = 0.0, delta_pitch: float = 0.0) -> None:
        """
        Turn camera in place.
        Positive delta_yaw turns right.
        Positive delta_pitch looks up.
        """
        self.heading = wrap_degrees(self.heading + delta_yaw)
        self._pitch = clamp(self._pitch + delta_pitch, -90.0, 90.0)

    def move_forward(self, distance_m: float = 20.0) -> None:
        """
        Approximate forward movement based on yaw.

        Important:
        This is only a rough geographic step, not true Street View graph navigation.
        Real Street View movement should use pano links / metadata if you want
        authentic GeoGuessr-like movement.
        """
        if distance_m == 0:
            return
        new_lon, new_lat, _ = _WGS84_GEOD.fwd(self._cur_lon, self._cur_lat, self.heading, distance_m)
        new_lon = ((new_lon + 180) % 360) - 180
        if not has_streetview(new_lat, new_lon):
            log_event(f"({new_lat},{new_lon}) has no street view")
            new_lat, new_lon = find_nearby_streetview(new_lat, new_lon, distance_m * 2)
            log_event(f"Moving to ({new_lat},{new_lon})")
        self._cur_lat = new_lat
        self._cur_lon = new_lon

    def guess(self, lat: float, lon: float) -> float:
        """ WGS84 geodesic distance in kilometers. """
        distance_m: float = _WGS84_GEOD.inv(lon, lat, self._tar_lon, self._tar_lat)[2]
        return round(distance_m / 1000, 3)

    def get_state(self) -> dict[str, Any]:
        return {
            "view_lat": self._cur_lat,
            "view_lon": self._cur_lon,
            "target_lat": self._tar_lat,
            "target_lon": self._tar_lon,
            "heading": self.heading,
        }
