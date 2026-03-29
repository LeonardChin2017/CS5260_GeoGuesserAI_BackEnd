
import requests


def test_stream_run():
    # Test the health endpoint
    response = requests.get("http://localhost:3001/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    # Test starting the agent
    response = requests.post("http://localhost:3001/api/agent/start")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    # Test agent status
    response = requests.get("http://localhost:3001/api/agent/status")
    assert response.status_code == 200
    assert response.json() == {"running": True}

    # Test streaming run endpoint
    run_payload = {
        "start_lat": 37.7749,
        "start_lon": -122.4194,
        "start_heading": 0.0,
        "max_iter": 1,
    }
    with requests.post(
        "http://localhost:3001/api/agent/stream-run",
        json=run_payload,
        stream=True,
        timeout=60,
    ) as response:
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")

        done_event_seen = False
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            print(f"Received line: {line}")
            if line.strip() == "event: done":
                done_event_seen = True
                break

        assert done_event_seen

    # Test stopping the agent
    response = requests.post("http://localhost:3001/api/agent/stop")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    # Test agent status again
    response = requests.get("http://localhost:3001/api/agent/status")
    assert response.status_code == 200
    assert response.json() == {"running": False}

def test_stream_analyze():
    # Ensure agent is running for streaming analyze endpoint.
    response = requests.post("http://localhost:3001/api/agent/start")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

    analyze_payload = {
        # Tiny valid PNG payload (1x1 transparent image) in raw base64.
        "screenshot": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5W5c8AAAAASUVORK5CYII=",
        "heading": 0.0,
        "max_iter": 1,
        "cur_iter": 0,
    }

    with requests.post(
        "http://localhost:3001/api/agent/stream-analyze",
        json=analyze_payload,
        stream=True,
        timeout=60,
    ) as response:
        assert response.status_code == 200
        assert response.headers.get("content-type", "").startswith("text/event-stream")

        done_event_seen = False
        data_event_seen = False
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            print(f"Received line: {line}")
            if line.startswith("data:"):
                data_event_seen = True
            if line.strip() == "event: done":
                done_event_seen = True
                break

        assert data_event_seen
        assert done_event_seen

    # Stop agent after test to avoid side effects between runs.
    response = requests.post("http://localhost:3001/api/agent/stop")
    assert response.status_code == 200
    assert response.json() == {"ok": True}

if __name__ == "__main__":
    test_stream_run()
    test_stream_analyze()
    print("All tests passed!")