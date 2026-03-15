# GeoGuesser Agent — Backend Architecture

## Overview

The backend is a FastAPI application (`app.py`) that serves as a **GeoGuessr AI Agent** — a multi-agent LangGraph pipeline that takes a Street View screenshot, runs 5 parallel specialist agents via Gemini, fuses the evidence, and places a guess.

---

## Technology Stack

| Component | Technology |
|---|---|
| Web framework | FastAPI + uvicorn (port 3001) |
| AI (vision) | Google Gemini (gemini-2.5-flash by default) |
| Graph pipeline | LangGraph (`StateGraph`) |
| Street View frames | Google Maps Street View Static API |

---

## File Structure

```
JobAIBackEnd/
├── app.py                         # Main FastAPI application
├── agent.py                       # Agent control and information kept 
├── game.py                        # Environment infomation; image capture; random streetview picking; game evaluation
├── util.py                        # Logging & API keys
├── graphs/
│   ├── state.py                   # GeoState TypedDict
│   ├── geoguessr_graph.py         # LangGraph pipeline definition
│   └── nodes/
│       ├── ingest.py              # Input validation node
│       ├── specialists.py         # 5 Gemini specialist nodes (parallel)
│       ├── fusion.py              # Fusion/planner node (decision making)
│       ├── gemini_vision.py       # Shared Gemini REST helper
│       └── stubs.py               # Test stubs (mock nodes for unit tests)
└── tests/
    ├── test_stage1_foundation.py  # Graph wiring + API endpoint tests
    └── test_gemini_specialists.py # Specialist node unit + integration tests

```

---

## GeoGuessr Agent Pipeline

### Graph Topology

```
ingest
  ├─→ text_language    ─┐
  ├─→ architecture     ─┤
  ├─→ climate_terrain  ─┼─→ fusion_planner → END
  ├─→ vegetation       ─┤
  └─→ road_infra       ─┘
       (parallel fan-out)     (fan-in)
```

The 5 specialist nodes run in **parallel** via LangGraph's fan-out. Each writes its own key into `specialist_outputs`; the `Annotated[dict, _merge_dicts]` reducer in `GeoState` merges them safely.

### State: `GeoState` (`graphs/state.py`)

```python
class GeoState(TypedDict):
    screenshot: str               # base64 data URL (input image)
    iteration: int                # current iteration (0-indexed)
    max_iterations: int           # budget (stop exploring after this)
    specialist_outputs: Annotated[dict, _merge_dicts]  # merged by parallel reducer
    belief_state: list            # ranked candidate locations [{country, lat, lon, confidence, evidence}]
    action: dict                  # {type: GUESS|ROTATE|MOVE, ...}
    final_guess: dict   # set when committing a GUESS
    error: str          # error from any node
```

### Nodes

#### `ingest_node` (`nodes/ingest.py`)
Validates screenshot presence. Sets `error` if missing.

#### Specialist Nodes (`nodes/specialists.py`)
Five focused Gemini vision calls, each returning structured JSON:

| Node | Key | Output fields |
|---|---|---|
| `text_language_node` | `text_language` | `detected_scripts`, `language_hints`, `place_names`, `confidence`, `evidence` |
| `architecture_node` | `architecture` | `building_styles`, `materials`, `urban_density`, `street_furniture`, `national_flags_or_symbols`, `confidence`, `evidence` |
| `climate_terrain_node` | `climate_terrain` | `climate_zone`, `terrain_type`, `sky_and_light`, `soil_color`, `road_surface`, `confidence`, `evidence` |
| `vegetation_node` | `vegetation` | `vegetation_type`, `biome`, `notable_species`, `season_hints`, `confidence`, `evidence` |
| `road_infra_node` | `road_infra` | `driving_side`, `road_markings`, `sign_shapes_colors`, `vehicle_types`, `road_quality`, `camera_rig_clues`, `confidence`, `evidence` |

All specialists use the shared `_run_specialist()` helper which handles JSON parsing and graceful fallback on API error.

**Evidence weighting hierarchy** (encoded in the fusion prompt):
1. Text & Language — highest weight (a script narrows to <20 countries)
2. Road Infrastructure — high weight (driving side halves the search space)
3. Architecture — medium (strong within a continent)
4. Climate & Terrain — medium (eliminates hemispheres and biomes)
5. Vegetation — medium (useful corroboration, weak alone)

#### `fusion_planner_node` (`nodes/fusion.py`)
Synthesises all 5 specialist outputs and decides:
- **GUESS** — if `top_confidence >= 0.75` or budget exhausted
- **ROTATE(degrees)** — rotate heading to reveal more clues
- **MOVE(forward)** — advance ~50m toward a visible landmark

Forces GUESS at `iteration >= max_iterations - 1`. Falls back to `(0, 0)` on API error.

#### `call_gemini_vision` (`nodes/gemini_vision.py`)
Shared Gemini REST helper. Retries up to 3× on HTTP 429 (rate limit) with 15s/30s/45s backoff. Parses JSON from markdown fences. Raises `ValueError` if API key is missing.

### Action Execution

`Game` tracks lat/lon/heading. `Agent.run()` applies ROTATE or MOVE and fetches the next Street View frame via the Static API.

```
ROTATE(degrees) → same position, new heading
MOVE(forward)   → advance 50m along current heading
```

---

## Agent Runner

Async background coroutine (`agent.run()`) that:

1. **Capture** — fetches Street View image
2. **Analyse** — invokes `geo_graph` (specialists + fusion) in a thread executor
3. **Decide** — if GUESS: compute score and stop; if ROTATE/MOVE: execute and loop

Updates `AGENT_STATE` throughout so the frontend polling endpoints stay current.

### Pipeline Steps (visible to frontend)

```
capture        → Capturing Street View frame
text_language  → Text & Language
architecture   → Architecture
climate_terrain → Climate & Terrain
vegetation     → Vegetation
road_infra     → Road & Infrastructure
reason         → Fusing evidence & planning
guess          → Placing guess
```

Each step gets a `status` (`pending | running | done`) and optionally a `detail` string (e.g., `"Cyrillic, Latin (85%)"` for text_language).

---

## API Endpoints

### GeoGuessr Agent (LangGraph mode)

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/agent/start` | Start a new agent run (new random target, launches LangGraph runner) |
| `POST` | `/api/agent/stop` | Abort the running agent |
| `GET` | `/api/agent/status` | Poll agent state (steps, game, errors) |
| `GET` | `/api/agent/frame` | Poll current Street View frame (base64 JPEG) |
| `POST` | `/api/agent/analyze` | Single-iteration LangGraph run on a provided screenshot |
| `POST` | `/api/agent/run` | Full multi-iteration autopilot loop (blocking, returns final result) |
| `GET` | `/api/agent/captures` | List saved capture files |

## LangGraph Agent Mode
- **Start**: `POST /api/agent/start`
- Backend fetches Street View frames itself (no browser needed)
- Backend runs all Gemini calls
- Frontend only polls `/api/agent/status` and `/api/agent/frame`
- Frontend shows backend JPEG (not Google Maps panorama) during run
- Panorama position is silently updated so it's synced when agent finishes

---

## Environment Variables

```env
GEMINI_API_KEY=         # Required for agent analysis
GEMINI_MODEL=           # Default: gemini-2.5-flash (use gemini-2.0-flash-lite for higher free-tier quota)
GOOGLE_MAPS_API_KEY=    # Required for Street View frame fetching
LOG_LEVEL=              # Default: WARNING
AGENT_MAX_ITERATIONS=   # Default: 5
AGENT_STEP_DELAY_MIN_SECONDS=  # Default: 0.8
AGENT_STEP_DELAY_MAX_SECONDS=  # Default: 1.8
```

---

## Known Limitations
- **Gemini free tier**: `gemini-2.5-flash` has a 20 req/day free limit. Each agent run uses 6 calls (5 specialists + 1 fusion). Use `gemini-2.0-flash-lite` (1500 req/day) or enable billing.
- **No real GeoGuessr integration**: Targets are randomly selected from 4 built-in demo locations.
