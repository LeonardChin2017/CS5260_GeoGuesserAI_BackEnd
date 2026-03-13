"""
GeoGuessr LangGraph pipeline.

Graph topology:
  ingest
    → [text_language, architecture, climate_terrain, vegetation, road_infra]  (parallel fan-out)
    → fusion_planner  (synthesises evidence, decides GUESS or EXPLORE)
    → END

The graph runs one iteration per invocation. The caller loops if the action
is ROTATE or MOVE, feeding the new screenshot back into the graph.
"""
from langgraph.graph import StateGraph, END
from graphs.state import GeoState
from graphs.nodes.stubs import ingest_node
from graphs.nodes.specialists import (
    text_language_node,
    architecture_node,
    climate_terrain_node,
    vegetation_node,
    road_infra_node,
)
from graphs.nodes.fusion import fusion_planner_node


def build_graph() -> StateGraph:
    graph = StateGraph(GeoState)

    graph.add_node("ingest", ingest_node)
    graph.add_node("text_language", text_language_node)
    graph.add_node("architecture", architecture_node)
    graph.add_node("climate_terrain", climate_terrain_node)
    graph.add_node("vegetation", vegetation_node)
    graph.add_node("road_infra", road_infra_node)
    graph.add_node("fusion_planner", fusion_planner_node)

    graph.set_entry_point("ingest")

    # ingest → all specialists in parallel (fan-out)
    for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
        graph.add_edge("ingest", specialist)

    # all specialists → fusion_planner (fan-in)
    for specialist in ["text_language", "architecture", "climate_terrain", "vegetation", "road_infra"]:
        graph.add_edge(specialist, "fusion_planner")

    graph.add_edge("fusion_planner", END)

    return graph.compile()


geo_graph = build_graph()
