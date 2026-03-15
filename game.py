import base64
from dataclasses import dataclass
from math import nan

import requests
from pyproj import Geod

from util import log_event, GOOGLE_MAPS_API_KEY

TEST_LOCATIONS = [
    {"lat": 35.6595, "lon": 139.7005},
    {"lat": -22.9711, "lon": -43.1822},
    {"lat": -33.9628, "lon": 18.4098},
    {"lat": 64.1466, "lon": -21.9426},
    {"lat": 1.2975414, "lon": 103.7779669}
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

    def reset(self, lat: float, lon: float, heading: float) -> None:
        self._cur_lat = self._tar_lat = lat
        self._cur_lon = self._tar_lon = lon
        self._pitch = 0.0
        self.heading = heading

    def set_to_random_street_view(self) -> None:
        """TODO pick randomly in the world"""
        target = TEST_LOCATIONS[4]
        self.reset(target["lat"], target["lon"], 0.0)

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

        TODO prevent moving to places having no street view
        """
        if distance_m == 0:
            return
        new_lon, new_lat, _ = _WGS84_GEOD.fwd(self._cur_lon, self._cur_lat, self.heading, distance_m)
        self._cur_lon = ((new_lon + 180) % 360) - 180
        self._cur_lat = new_lat

    def guess(self, lat: float, lon: float) -> float:
        """ WGS84 geodesic distance in kilometers. """
        distance_m: float = _WGS84_GEOD.inv(lon, lat, self._tar_lon, self._tar_lat)[2]
        return round(distance_m / 1000, 3)


if __name__ == "__main__":
    pass
