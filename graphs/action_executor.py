"""
Action executor — translates ROTATE/MOVE decisions into a new Street View screenshot.

ROTATE(degrees): same position, rotate heading by `degrees`
MOVE(forward):   advance ~50m along current heading, same heading

Street View frames are fetched via the Google Maps Street View Static API and
returned as base64 data URLs ready for the next graph iteration.
"""
import base64
import math

import requests

# How far (metres) a MOVE action advances the viewpoint
MOVE_DISTANCE_METRES = 50


class GameView:
    """Tracks the current Street View position and heading."""

    def __init__(self, lat: float, lon: float, heading: int = 0):
        self.lat = lat
        self.lon = lon
        self.heading = heading % 360

    def rotate(self, degrees: int) -> "GameView":
        return GameView(self.lat, self.lon, (self.heading + degrees) % 360)

    def move_forward(self, metres: int = MOVE_DISTANCE_METRES) -> "GameView":
        """Move `metres` in the direction of current heading."""
        heading_rad = math.radians(self.heading)
        delta_lat = (metres * math.cos(heading_rad)) / 111_000
        delta_lon = (metres * math.sin(heading_rad)) / (
            111_000 * math.cos(math.radians(self.lat))
        )
        return GameView(self.lat + delta_lat, self.lon + delta_lon, self.heading)

    def to_dict(self) -> dict:
        return {"lat": self.lat, "lon": self.lon, "heading": self.heading}

    @classmethod
    def from_dict(cls, d: dict) -> "GameView":
        return cls(d["lat"], d["lon"], d.get("heading", 0))


def fetch_streetview_screenshot(view: GameView, api_key: str, size: str = "640x640") -> str:
    """
    Fetch a Street View Static API frame and return a base64 data URL.
    Raises ValueError if api_key is missing.
    Raises requests.HTTPError on API errors.
    """
    if not api_key:
        raise ValueError("GOOGLE_MAPS_API_KEY is not set")

    url = "https://maps.googleapis.com/maps/api/streetview"
    params = {
        "size": size,
        "location": f"{view.lat},{view.lon}",
        "heading": view.heading,
        "fov": 90,
        "pitch": 0,
        "key": api_key,
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()

    b64 = base64.b64encode(resp.content).decode()
    return f"data:image/jpeg;base64,{b64}"


def execute_action(action: dict, view: GameView, api_key: str) -> tuple[str, GameView]:
    """
    Apply action to the current view and fetch the resulting Street View frame.

    Returns:
        (screenshot_b64, new_view)

    Raises on unsupported action types or API failures.
    """
    action_type = action.get("type", "").upper()

    if action_type == "ROTATE":
        degrees = int(action.get("degrees", 90))
        new_view = view.rotate(degrees)
    elif action_type == "MOVE":
        new_view = view.move_forward()
    else:
        raise ValueError(f"execute_action called with non-explore action: {action_type}")

    screenshot = fetch_streetview_screenshot(new_view, api_key)
    return screenshot, new_view
