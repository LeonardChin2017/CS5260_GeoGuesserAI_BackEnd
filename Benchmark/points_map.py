"""
To visualize all points in LOCATION_DATABASE on http://127.0.0.1:5000,
Run `PYTHONPATH=. python Benchmark/points_map.py`
"""

import json

from flask import Flask, render_template_string

from util import GOOGLE_MAPS_API_KEY, LOCATION_DATABASE

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Google Map Markers</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        html, body {
            height: 100%;
            margin: 0;
            padding: 0;
        }
        #map {
            height: 100%;
            width: 100%;
        }
    </style>
</head>
<body>
    <div id="map"></div>

    <script>
        const locations = {{ locations | safe }};

        function initMap() {
            const defaultCenter = locations.length > 0
                ? { lat: locations[0][0], lng: locations[0][1] }
                : { lat: 0, lng: 0 };

            const map = new google.maps.Map(document.getElementById("map"), {
                zoom: 4,
                center: defaultCenter,
            });

            const bounds = new google.maps.LatLngBounds();

            for (const [lat, lng] of locations) {
                const position = { lat, lng };

                new google.maps.Marker({
                    position,
                    map,
                });

                bounds.extend(position);
            }

            if (locations.length > 0) {
                map.fitBounds(bounds);

                // Prevent zooming in too far when there is only one marker
                google.maps.event.addListenerOnce(map, "bounds_changed", function () {
                    if (this.getZoom() > 14) {
                        this.setZoom(14);
                    }
                });
            }
        }
    </script>

    <script async defer
        src="https://maps.googleapis.com/maps/api/js?key={{ api_key }}&callback=initMap">
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML,
        locations=json.dumps(LOCATION_DATABASE),
        api_key=GOOGLE_MAPS_API_KEY,
    )


if __name__ == "__main__":
    app.run(debug=True)