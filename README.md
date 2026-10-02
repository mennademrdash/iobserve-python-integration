# I-Observe Python Integration

Standalone Python client for connecting a Frigate or RTSP stream to an existing
I-Observe server. The client only transports frames; detection, liveness,
recognition, scene analysis, VLM processing, event persistence, and evidence
generation remain on the I-Observe server.

## Install

```bash
pip install -r requirements.txt
```

## Configure

Never commit credentials. Provide them through environment variables:

```powershell
$env:I_OBSERVE_URL="http://your-iobserve-server:8000"
$env:I_OBSERVE_TOKEN="<YOUR_TOKEN>"
```

## Connect a Frigate/RTSP stream

```python
import os
from iobserve_integration import IObserve

api = IObserve(
    base_url=os.environ["I_OBSERVE_URL"],
    token=os.environ["I_OBSERVE_TOKEN"],
)

connection = api.connect_stream(
    stream_url="rtsp://frigate:8554/front_door",
    sample_fps=2,
)

try:
    for result in connection.iter_results():
        for face in result.detections:
            print(face.identity, face.score, face.live)
        print("VLM:", result.vlm)
        print("Objects:", result.objects)
finally:
    connection.stop()
    api.close()
```

For Frigate go2rtc/restream, use the corresponding RTSP URL, commonly:

```text
rtsp://<frigate-host>:8554/<camera-name>
```

## Server API used

- `POST /api/live/start`
- `POST /api/live/observe`
- `POST /api/live/stop`
- `GET /api/events`

The integration does not contain a duplicate or fallback AI pipeline.
