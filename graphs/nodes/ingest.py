"""Ingest node — validates screenshot is present before the specialist fan-out."""
from graphs.state import GeoState


def ingest_node(state: GeoState) -> dict:
    """Validate screenshot is present; pass through."""
    if not state.get("screenshot"):
        return {"error": "No screenshot provided"}
    return {"error": '', "message": "Screenshot ingested. Processing..."}
