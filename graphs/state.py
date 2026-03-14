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

    # How many exploration iterations have been used so far
    iteration: int

    # Budget: stop exploring and commit guess after this many iterations
    max_iterations: int

    # Outputs from each specialist agent, keyed by agent name.
    # Annotated with _merge_dicts so parallel nodes can each write their own key
    # without LangGraph raising InvalidUpdateError.
    specialist_outputs: Annotated[dict, _merge_dicts]

    # Ranked list of candidate locations
    belief_state: list

    # The next action to take
    action: dict

    # Full history of actions taken so far TODO delete
    action_history: list

    # Set when the agent commits a final guess
    final_guess: Optional[dict]

    # Optional error message from any node
    error: Optional[str]
