from typing import TypedDict, Optional, Annotated


def _merge_dicts(a: dict, b: dict) -> dict:
    """Reducer: merge two dicts. Used for specialist_outputs written by parallel nodes."""
    merged = dict(a or {})
    merged.update(b or {})
    return merged


class GeoState(TypedDict):
    """Shared state passed between all nodes in the GeoGuessr LangGraph pipeline."""

    # Input: base64-encoded screenshot (data URL or raw base64 JPEG)
    screenshot: str

    # The mode of operation (e.g., "analyze", "run")
    mode : str

    # The current iterations number (0-indexed)
    iteration: int

    # Budget: stop exploring and commit guess after this many iterations
    max_iterations: int

    # "CONTINUE" or "STOP", set by iteration_guard and read by route_exploration_loop
    loop_decision: str  

    # Outputs from each specialist agent, keyed by agent name.
    # Annotated with _merge_dicts so parallel nodes can each write their own key
    # without LangGraph raising InvalidUpdateError.
    specialist_outputs: Annotated[dict, _merge_dicts]

    # Ranked list of candidate locations
    belief_state: list

    # The next action to take
    action: dict

    # Set when the agent commits a final guess
    final_guess: dict

    # Optional error message from any node
    error: str

    # Optional human-readable message for frontend (e.g., "Rotating 90 degrees right")
    message: Optional[str]
