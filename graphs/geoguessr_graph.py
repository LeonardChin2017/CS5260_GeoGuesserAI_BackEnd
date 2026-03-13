"""
GeoGuessr LangGraph pipeline.

Stage 1: All specialist nodes are stubs. Graph structure and state flow are wired up.
Later stages will swap stubs for real LLM-powered nodes.

Graph topology:
  ingest
    → [text_language, architecture, climate_terrain, vegetation, road_infra]  (parallel fan-out)
    → fusion_planner
    → END  (action = GUESS or ROTATE/MOVE; caller loops if exploring)
"""
from langgraph.graph import StateGraph, END
from graphs.state import GeoState
from graphs.nodes.stubs import ingest_node, fusion_planner_stub
from graphs.nodes.specialists import (
    text_language_node,
    architecture_node,
    climate_terrain_node,
    vegetation_node,
    road_infra_node,
)

# ---------------------------------------------------------------------------
# Reducer: merge specialist_outputs dicts across parallel branches
# ---------------------------------------------------------------------------
from typing import Annotated
import operator


def _merge_specialist_outputs(a: dict, b: dict) -> dict:
    """Merge two specialist_outputs dicts (used as a state reducer)."""
    merged = dict(a or {})
    merged.update(b or {})
    return merged


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(GeoState)

    # Nodes
    graph.add_node("ingest", ingest_node)
    graph.add_node("text_language", text_language_node)
    graph.add_node("architecture", architecture_node)
    graph.add_node("climate_terrain", climate_terrain_node)
    graph.add_node("vegetation", vegetation_node)
    graph.add_node("road_infra", road_infra_node)
    graph.add_node("fusion_planner", fusion_planner_stub)

    # Entry point
    graph.set_entry_point("ingest")

    # ingest → all specialists (fan-out)
    for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
        graph.add_edge("ingest", specialist)

    # all specialists → fusion_planner (fan-in)
    for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
        graph.add_edge(specialist, "fusion_planner")

    # fusion_planner → END
    graph.add_edge("fusion_planner", END)

    return graph.compile()


# Singleton compiled graph
geo_graph = build_graph()
