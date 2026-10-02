#!/usr/bin/env python3
"""I‑Observe Backend Integration Kit
====================================

Single-file SDK for backend developers who need to integrate with
the **I‑Observe** face-recognition / video-intelligence pipeline.

How to use
----------
1.  Install the runtime dependencies::

        pip install requests opencv-python

2.  Set credentials in your environment or pass them to the constructor::

        export I_OBSERVE_URL=http://localhost:8000
        export I_OBSERVE_TOKEN=sk-or-v1-...
        # OR
        export I_OBSERVE_USERNAME=demo
        export I_OBSERVE_PASSWORD=demo123456

3.  Import and use::

        from iobserve_integration import IObserve

        api = IObserve(base_url="http://iobserve:8000", token="...")
        connection = api.connect_stream("rtsp://frigate:8554/front_door")
        for result in connection.iter_results():
            print(result.detections, result.vlm)

Configuration
-------------
Constructor arguments (all optional):

``base_url``    I‑Observe server.  Default ``http://localhost:8000``
``token``       Permanent ``Authorization: Bearer`` token (JWT or API_TOKEN).
                If omitted the client tries ``I_OBSERVE_TOKEN`` / ``API_TOKEN``
                env vars, then falls back to login via username/password.
``username``    Env fallback ``I_OBSERVE_USERNAME``.
``password``    Env fallback ``I_OBSERVE_PASSWORD``.
``timeout``     HTTP request timeout in seconds (default 30).

When ``token`` is provided it is used directly.  Otherwise the client
authenticates with username/password at construction time and stores the
returned JWT for the lifetime of the ``IObserve`` instance.

API coverage (endpoints)
------------------------
POST /api/auth/login          →  login()
POST /api/auth/register       →  register()
GET  /api/auth/me             →  whoami()
POST /api/auth/logout         →  logout()
GET  /api/status              →  system_status()
GET  /api/counters            →  counters()
GET  /api/events              →  search_events()
GET  /api/events/{id}         →  get_event()
GET  /api/live/status         →  live_status()
POST /api/live/start          →  live_start()
POST /api/live/stop           →  live_stop()
POST /api/camera/detect       →  detect_frame()
GET  /api/live/recordings     →  live_recordings()
POST /api/vlm/test            →  vlm_test()
POST /enroll                  →  enroll()
POST /query                   →  ask()
GET  /health                  →  health()
"""

from __future__ import annotations

import base64
import io
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any
from urllib.parse import urlencode

__version__ = "2.1.0"

# ---------------------------------------------------------------------------
# Data models  (mirror the server's Pydantic / response payloads)
# ---------------------------------------------------------------------------


@dataclass
class ComponentHealth:
    """Health of one backend component (PostgreSQL, Redis, Face stack, …)."""
    ok: bool
    detail: str = ""


@dataclass
class SystemStatus:
    """GET /api/status full response."""
    api: ComponentHealth | None = None
    postgres: ComponentHealth | None = None
    redis: ComponentHealth | None = None
    face_index: ComponentHealth | None = None
    storage: ComponentHealth | None = None
    face: ComponentHealth | None = None
    objects: ComponentHealth | None = None
    liveness: ComponentHealth | None = None
    videomae: ComponentHealth | None = None
    vlm: ComponentHealth | None = None
    identities: list[dict] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    last_run: Any = None
    job_running: bool = False
    videomae_last: int = 0

    @staticmethod
    def from_dict(d: dict) -> "SystemStatus":
        def _comp(key: str) -> ComponentHealth | None:
            v = d.get(key)
            if not isinstance(v, dict):
                return None
            return ComponentHealth(ok=bool(v.get("ok")), detail=str(v.get("detail", "")))
        return SystemStatus(
            api=_comp("api"), postgres=_comp("postgres"), redis=_comp("redis"),
            face_index=_comp("face_index"), storage=_comp("storage"), face=_comp("face"),
            objects=_comp("objects"), liveness=_comp("liveness"), videomae=_comp("videomae"),
            vlm=_comp("vlm"),
            identities=list(d.get("identities", [])),
            config=dict(d.get("config", {})),
            counts=dict(d.get("counts", {})),
            last_run=d.get("last_run"),
            job_running=bool(d.get("job_running")),
            videomae_last=int(d.get("videomae_last", 0)),
        )


@dataclass
class Counters:
    """GET /api/counters response."""
    events: int = 0
    face_detections: int = 0
    identified: int = 0
    unknown: int = 0
    live: int = 0
    spoof_rejected: int = 0
    evidence_images: int = 0
    evidence_clips: int = 0
    identities: int = 0
    max_similarity: float = 0.0
    latest_action: str | None = None
    latest_action_score: float | None = None
    latest_action_ts: str | None = None

    @staticmethod
    def from_dict(d: dict) -> "Counters":
        return Counters(
            events=int(d.get("events", 0)),
            face_detections=int(d.get("face_detections", 0)),
            identified=int(d.get("identified", 0)),
            unknown=int(d.get("unknown", 0)),
            live=int(d.get("live", 0)),
            spoof_rejected=int(d.get("spoof_rejected", 0)),
            evidence_images=int(d.get("evidence_images", 0)),
            evidence_clips=int(d.get("evidence_clips", 0)),
            identities=int(d.get("identities", 0)),
            max_similarity=float(d.get("max_similarity", 0.0)),
            latest_action=d.get("latest_action"),
            latest_action_score=_opt_float(d.get("latest_action_score")),
            latest_action_ts=d.get("latest_action_ts"),
        )


@dataclass
class Event:
    """An event from the timeline."""
    id: str
    track_id: int = 0
    ts: float = 0.0
    wall_ts: float = 0.0
    recorded_at: str = ""
    timestamp_hms: str = ""
    event_type: str = ""
    shirt_color: str | None = None
    identity: str | None = None
    identity_score: float = 0.0
    action: str | None = None
    action_score: float = 0.0
    vlm: str | None = None
    source: str = ""
    session: str | None = None
    objects: list = field(default_factory=list)
    rules: list = field(default_factory=list)
    image_url: str | None = None
    clip_url: str | None = None
    bbox: list | None = None
    evidence: dict = field(default_factory=dict)

    @staticmethod
    def from_dict(d: dict) -> "Event":
        return Event(
            id=str(d.get("id", "")),
            track_id=int(d.get("track_id", 0)),
            ts=float(d.get("ts", 0.0)),
            wall_ts=float(d.get("wall_ts", 0.0)),
            recorded_at=str(d.get("recorded_at", "")),
            timestamp_hms=str(d.get("timestamp_hms", "")),
            event_type=str(d.get("event_type", "")),
            shirt_color=d.get("shirt_color"),
            identity=d.get("identity"),
            identity_score=float(d.get("identity_score", 0.0)),
            action=d.get("action"),
            action_score=float(d.get("action_score", 0.0)),
            vlm=d.get("vlm"),
            source=str(d.get("source", "")),
            session=d.get("session"),
            objects=list(d.get("objects", [])),
            rules=list(d.get("rules", [])),
            image_url=d.get("image_url"),
            clip_url=d.get("clip_url"),
            bbox=d.get("bbox"),
            evidence=d.get("evidence", {}),
        )


@dataclass
class Detection:
    """Face detection + recognition result for one face."""
    index: int = 0
    bbox: list | None = None
    det_score: float = 0.0
    landmarks: list = field(default_factory=list)
    identity: str = "unknown"
    score: float = 0.0
    live: bool = False
    liveness_score: float = 0.0
    spoof_score: float = 0.0
    liveness_error: str | None = None
    error: str | None = None
    track_id: int | None = None

    @staticmethod
    def from_dict(d: dict) -> "Detection":
        return Detection(
            index=int(d.get("index", 0)),
            bbox=d.get("bbox"),
            det_score=float(d.get("det_score", 0.0)),
            landmarks=list(d.get("landmarks", [])),
            identity=str(d.get("identity", "unknown")),
            score=float(d.get("score", 0.0)),
            live=bool(d.get("live")),
            liveness_score=float(d.get("liveness_score", 0.0)),
            spoof_score=float(d.get("spoof_score", 0.0)),
            liveness_error=d.get("liveness_error"),
            error=d.get("error"),
            track_id=d.get("track_id"),
        )


@dataclass
class DetectResponse:
    """POST /api/camera/detect full response."""
    detections: list[Detection] = field(default_factory=list)
    face_count: int = 0
    objects: list[dict] = field(default_factory=list)
    inference_ms: float = 0.0
    events_recorded: int = 0
    session: str | None = None
    recording: bool = False
    observations: list[dict] = field(default_factory=list)
    object_error: str | None = None
    recording_error: str | None = None
    frame_size: list[int] = field(default_factory=list)
    scene: dict = field(default_factory=dict)
    vlm: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def from_dict(d: dict) -> "DetectResponse":
        return DetectResponse(
            detections=[Detection.from_dict(x) for x in d.get("detections", [])],
            face_count=int(d.get("face_count", 0)),
            objects=list(d.get("objects", [])),
            inference_ms=float(d.get("inference_ms", 0.0)),
            events_recorded=int(d.get("events_recorded", 0)),
            session=d.get("session"),
            recording=bool(d.get("recording")),
            observations=list(d.get("observations", [])),
            object_error=d.get("object_error"),
            recording_error=d.get("recording_error"),
            frame_size=list(d.get("frame_size", [])),
            scene=dict(d.get("scene") or {}),
            vlm=dict(d.get("vlm") or {}),
            raw=dict(d),
        )


@dataclass
class StreamStats:
    """Live counters for one external Frigate/RTSP connection."""

    frames_read: int = 0
    frames_sent: int = 0
    results_received: int = 0
    reconnects: int = 0
    started_at: float = 0.0
    last_result_at: float | None = None


class StreamConnection:
    """Background bridge from an OpenCV-compatible stream into I-Observe.

    This class transports JPEG frames only. The existing I-Observe server performs
    all face, liveness, identity, scene, VLM, event and evidence processing.
    """

    def __init__(self, api: "IObserve", stream_url: str | int, *, sample_fps: float = 2.0,
                 jpeg_quality: int = 85, reconnect_delay: float = 2.0,
                 open_timeout: float = 10.0, read_timeout: float = 10.0,
                 record_events: bool = True, on_result=None, on_error=None,
                 max_frames: int | None = None):
        if sample_fps <= 0:
            raise ValueError("sample_fps must be greater than zero")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.api = api
        self.stream_url = stream_url
        self.sample_fps = float(sample_fps)
        self.jpeg_quality = int(jpeg_quality)
        self.reconnect_delay = max(0.1, float(reconnect_delay))
        self.open_timeout = max(0.1, float(open_timeout))
        self.read_timeout = max(0.1, float(read_timeout))
        self.record_events = bool(record_events)
        self.on_result = on_result
        self.on_error = on_error
        self.max_frames = max_frames
        self.stats = StreamStats()
        self.session: str | None = None
        self.last_result: DetectResponse | None = None
        self.last_error: Exception | None = None
        self._results: queue.Queue[DetectResponse] = queue.Queue(maxsize=100)
        self._stop = threading.Event()
        self._connected = threading.Event()
        self._done = threading.Event()
        self._capture = None
        self._thread = threading.Thread(target=self._run, name="iobserve-stream-bridge", daemon=True)

    @property
    def connected(self) -> bool:
        return self._connected.is_set() and not self._done.is_set()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def start(self) -> "StreamConnection":
        if self._thread.is_alive():
            return self
        # Import native OpenCV on the caller thread. On Windows, loading its FFmpeg DLLs
        # for the first time inside a worker can exceed the connection timeout.
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Stream capture requires: pip install opencv-python") from exc
        self.stats.started_at = time.time()
        if self.record_events:
            state = self.api.live_start()
            self.session = state.get("session")
        self._thread.start()
        return self

    def wait_until_connected(self, timeout: float = 10.0) -> bool:
        return self._connected.wait(timeout)

    def get_result(self, timeout: float | None = None) -> DetectResponse:
        """Return the next real analysis result received from I-Observe."""
        return self._results.get(timeout=timeout)

    def iter_results(self, timeout: float = 1.0):
        while not self.done or not self._results.empty():
            try:
                yield self.get_result(timeout=timeout)
            except queue.Empty:
                continue

    def events(self, limit: int = 100) -> list["Event"]:
        if not self.session:
            return self.api.search_events(limit=limit)
        # Date queries use PostgreSQL wall-clock ordering. The generic event endpoint
        # orders by clip offset, where an older long recording can hide a new session.
        session_date = time.strftime("%Y-%m-%d", time.localtime(self.stats.started_at))
        events = self.api.search_events(limit=min(max(limit, 1), 1000), date=session_date)
        return [event for event in events if event.session == self.session]

    def stop(self, timeout: float = 15.0) -> dict | None:
        self._stop.set()
        if self._capture is not None:
            self._capture.release()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout)
        return self.api.live_stop() if self.record_events else None

    def join(self, timeout: float | None = None) -> bool:
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def raise_if_failed(self) -> None:
        if self.last_error is not None:
            raise self.last_error

    def _report_error(self, exc: Exception) -> None:
        self.last_error = exc
        if self.on_error:
            self.on_error(exc)

    def _open_capture(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Stream capture requires: pip install opencv-python") from exc
        capture = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.open_timeout * 1000))
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout * 1000))
        if not capture.open(self.stream_url, cv2.CAP_FFMPEG):
            capture.release()
            raise ConnectionError(f"Could not open external stream: {self.stream_url}")
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _publish_result(self, result: DetectResponse) -> None:
        self.last_result = result
        self.stats.results_received += 1
        self.stats.last_result_at = time.time()
        if self._results.full():
            try:
                self._results.get_nowait()
            except queue.Empty:
                pass
        self._results.put_nowait(result)
        if self.on_result:
            self.on_result(result)

    def _run(self) -> None:
        import cv2

        next_send = 0.0
        finite_source = isinstance(self.stream_url, (str, os.PathLike)) and Path(self.stream_url).is_file()
        try:
            while not self._stop.is_set():
                try:
                    self._capture = self._open_capture()
                    self._connected.set()
                    while not self._stop.is_set():
                        ok, frame = self._capture.read()
                        if not ok:
                            if finite_source:
                                self._stop.set()
                            break
                        self.stats.frames_read += 1
                        now = time.monotonic()
                        if now < next_send:
                            continue
                        next_send = now + 1.0 / self.sample_fps
                        ok, encoded = cv2.imencode(
                            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                        )
                        if not ok:
                            self._report_error(RuntimeError("Could not JPEG-encode stream frame"))
                            continue
                        jpeg = encoded.tobytes()
                        result = (
                            self.api.observe_frame(jpeg, session=self.session)
                            if self.record_events
                            else self.api.detect_frame(jpeg)
                        )
                        self.stats.frames_sent += 1
                        self._publish_result(result)
                        if self.max_frames is not None and self.stats.frames_sent >= self.max_frames:
                            self._stop.set()
                            break
                except Exception as exc:
                    self._report_error(exc)
                finally:
                    if self._capture is not None:
                        self._capture.release()
                    self._capture = None
                if self._stop.is_set() or finite_source:
                    break
                self.stats.reconnects += 1
                self._stop.wait(self.reconnect_delay)
        finally:
            self._done.set()


@dataclass
class Recording:
    """One live-camera recording session."""
    session: str = ""
    frames: int = 0
    started: str = ""
    ended: str = ""
    duration_s: float = 0.0
    preview: str = ""
    recording_now: bool = False

    @staticmethod
    def from_dict(d: dict) -> "Recording":
        return Recording(
            session=str(d.get("session", "")),
            frames=int(d.get("frames", 0)),
            started=str(d.get("started", "")),
            ended=str(d.get("ended", "")),
            duration_s=float(d.get("duration_s", 0.0)),
            preview=str(d.get("preview", "")),
            recording_now=bool(d.get("recording_now")),
        )


@dataclass
class QueryResult:
    """POST /query natural-language response."""
    answer: str = ""
    identity: str | None = None
    track_id: int | None = None
    window: dict = field(default_factory=dict)
    total_events: int = 0
    evidence: list = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "QueryResult":
        return QueryResult(
            answer=str(d.get("answer", "")),
            identity=d.get("identity"),
            track_id=d.get("track_id"),
            window=dict(d.get("window", {})),
            total_events=int(d.get("total_events", 0)),
            evidence=list(d.get("evidence", [])),
        )


@dataclass
class EnrollResult:
    """POST /enroll response."""
    identity: str = ""
    point_id: str = ""
    point_ids: list = field(default_factory=list)
    dimension: int = 0
    images_used: int = 0
    rejected: list = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict) -> "EnrollResult":
        return EnrollResult(
            identity=str(d.get("identity", "")),
            point_id=str(d.get("point_id", "")),
            point_ids=list(d.get("point_ids", [])),
            dimension=int(d.get("dimension", 0)),
            images_used=int(d.get("images_used", 0)),
            rejected=list(d.get("rejected", [])),
        )


@dataclass
class VlmTestResult:
    """POST /api/vlm/test response."""
    ok: bool = False
    provider: str = ""
    model: str = ""
    answer: str = ""
    elapsed_s: float = 0.0
    error: str | None = None

    @staticmethod
    def from_dict(d: dict) -> "VlmTestResult":
        return VlmTestResult(
            ok=bool(d.get("ok")),
            provider=str(d.get("provider", "")),
            model=str(d.get("model", "")),
            answer=str(d.get("answer", "")),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
            error=d.get("error"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _opt_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class IObserveError(Exception):
    """Raised on non-2xx API responses."""
    def __init__(self, status: int, detail: str, body: Any = None):
        self.status = status
        self.detail = detail
        self.body = body
        super().__init__(f"[{status}] {detail}")


class IObserve:
    """Synchronous client for the I‑Observe REST API.

    Usage::

        api = IObserve()
        s = api.system_status()
        ev = api.search_events(limit=20)
        for e in ev:
            print(e.recorded_at, e.identity, e.event_type)
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        timeout: int = 30,
    ):
        import requests as _req
        self._requests = _req
        self._session = _req.Session()
        self._request_lock = threading.Lock()

        self.base_url = (base_url or os.getenv("I_OBSERVE_URL") or
                         os.getenv("IOBSERVE_URL") or "http://localhost:8000").rstrip("/")
        self.timeout = timeout
        self._token: str | None = None
        self._username: str | None = None

        # 1 – explicit token
        if token:
            self._token = token
        # 2 – env token
        elif os.getenv("I_OBSERVE_TOKEN"):
            self._token = os.getenv("I_OBSERVE_TOKEN")
        elif os.getenv("API_TOKEN"):
            self._token = os.getenv("API_TOKEN")
        # 3 – username/password
        else:
            u = username or os.getenv("I_OBSERVE_USERNAME") or ""
            p = password or os.getenv("I_OBSERVE_PASSWORD") or ""
            if u and p:
                self.login(u, p)

        if self._token:
            self._session.headers["Authorization"] = f"Bearer {self._token}"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        data: Any = None,
        files: Any = None,
        params: dict | None = None,
    ) -> Any:
        import json as _json
        url = self._url(path)
        kwargs = {"timeout": self.timeout, "params": params}
        if files and data:
            kwargs["data"] = data
            kwargs["files"] = files
        elif json is not None:
            kwargs["json"] = json
        elif data is not None:
            kwargs["data"] = data

        try:
            with self._request_lock:
                resp = self._session.request(method, url, **kwargs)
            body = resp.text
        except self._requests.exceptions.ConnectionError as exc:
            raise IObserveError(0, f"Cannot reach {url}: {exc}") from exc
        except self._requests.exceptions.Timeout as exc:
            raise IObserveError(0, f"Request to {url} timed out") from exc

        payload = None
        if body and body.startswith("{"):
            try:
                payload = _json.loads(body)
            except _json.JSONDecodeError:
                pass

        if resp.status_code >= 300:
            detail = ""
            if isinstance(payload, dict):
                detail = payload.get("detail") or payload.get("error") or ""
            raise IObserveError(resp.status_code, detail or body[:300], payload)

        return payload if payload is not None else body

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def login(self, username: str, password: str) -> dict:
        """Authenticate and store the JWT for subsequent requests."""
        body = self._request("POST", "/api/auth/login", json={
            "username": username, "password": password,
        })
        self._token = body.get("token", "")
        self._username = body.get("username", "")
        self._session.headers["Authorization"] = f"Bearer {self._token}"
        return body

    def register(self, username: str, password: str) -> dict:
        """Create a new account and sign in."""
        body = self._request("POST", "/api/auth/register", json={
            "username": username, "password": password,
        })
        self._token = body.get("token", "")
        self._username = body.get("username", "")
        self._session.headers["Authorization"] = f"Bearer {self._token}"
        return body

    def whoami(self) -> dict:
        """Return the current user info."""
        return self._request("GET", "/api/auth/me")

    def logout(self) -> dict:
        """Clear the server-side session cookie."""
        return self._request("POST", "/api/auth/logout")

    # ------------------------------------------------------------------
    # System
    # ------------------------------------------------------------------

    def system_status(self) -> SystemStatus:
        """Full health and configuration of every component."""
        body = self._request("GET", "/api/status")
        return SystemStatus.from_dict(body)

    def counters(self) -> Counters:
        """Aggregated event/face/identity counters."""
        body = self._request("GET", "/api/counters")
        return Counters.from_dict(body)

    def health(self) -> dict:
        """Simple liveness check (no auth required)."""
        return self._request("GET", "/health")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def search_events(
        self,
        *,
        limit: int = 100,
        identity: str | None = None,
        color: str | None = None,
        date: str | None = None,
        start: float | None = None,
        end: float | None = None,
    ) -> list[Event]:
        """Search the event timeline.  Returns newest events first."""
        params: dict[str, Any] = {"limit": max(1, min(limit, 1000))}
        if identity:
            params["identity"] = identity
        if color:
            params["color"] = color
        if date:
            params["date"] = date
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end
        body = self._request("GET", "/api/events", params=params)
        return [Event.from_dict(e) for e in body.get("events", [])]

    def get_event(self, event_id: str) -> Event:
        """Full detail for one event including evidence payload."""
        body = self._request("GET", f"/api/events/{event_id}")
        return Event.from_dict(body)

    # ------------------------------------------------------------------
    # Live Camera
    # ------------------------------------------------------------------

    def live_status(self) -> dict:
        """Current live-camera recording state."""
        return self._request("GET", "/api/live/status")

    def live_start(self) -> dict:
        """Begin a live recording session."""
        return self._request("POST", "/api/live/start")

    def live_stop(self) -> dict:
        """Stop the current live recording session."""
        return self._request("POST", "/api/live/stop")

    def detect_frame(self, image: bytes | IO[bytes]) -> DetectResponse:
        """Send one camera/JPEG frame for detection.

        ``image`` can be raw JPEG bytes or a file-like object.  Returns
        detections, face_count, YOLOv8n objects and session info.
        """
        data = image.read() if hasattr(image, "read") else image
        blob = io.BytesIO(data) if isinstance(data, bytes) else data
        files = {"image": ("frame.jpg", blob, "image/jpeg")}
        form = {"annotate": "false"}
        body = self._request("POST", "/api/camera/detect",
                             data=form, files=files)
        return DetectResponse.from_dict(body)

    def observe_frame(self, image: bytes | IO[bytes], session: str | None = None) -> DetectResponse:
        """Analyze and synchronously persist one external-stream frame.

        This calls the existing ``/api/live/observe`` pipeline, including PostgreSQL
        insert and server-side read-back verification, before returning.
        """
        data = image.read() if hasattr(image, "read") else image
        blob = io.BytesIO(data) if isinstance(data, bytes) else data
        files = {"image": ("frame.jpg", blob, "image/jpeg")}
        form = {"session": session or "external_stream"}
        body = self._request("POST", "/api/live/observe", data=form, files=files)
        return DetectResponse.from_dict(body)

    def connect_stream(
        self,
        stream_url: str | int,
        *,
        sample_fps: float = 2.0,
        on_result=None,
        on_error=None,
        record_events: bool = True,
        wait_timeout: float = 10.0,
        **options,
    ) -> StreamConnection:
        """Connect Frigate/RTSP to the existing I-Observe analysis pipeline.

        The URL may be RTSP, a Frigate restream URL, a camera index, or a local
        video path for deterministic tests. No AI runs in this integration file.
        """
        connection = StreamConnection(
            self,
            stream_url,
            sample_fps=sample_fps,
            on_result=on_result,
            on_error=on_error,
            record_events=record_events,
            **options,
        ).start()
        if not connection.wait_until_connected(wait_timeout):
            connection.stop()
            connection.raise_if_failed()
            raise TimeoutError(f"Timed out opening external stream: {stream_url}")
        return connection

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "IObserve":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def live_recordings(self) -> list[Recording]:
        """All recorded live-camera sessions."""
        body = self._request("GET", "/api/live/recordings")
        return [Recording.from_dict(r) for r in body.get("recordings", [])]

    # ------------------------------------------------------------------
    # VLM (Vision Language Model)
    # ------------------------------------------------------------------

    def vlm_test(self, language: str = "") -> VlmTestResult:
        """Run one real VLM request to verify the provider works."""
        body = self._request("POST", "/api/vlm/test",
                             json={"language": language})
        return VlmTestResult.from_dict(body)

    # ------------------------------------------------------------------
    # Enroll
    # ------------------------------------------------------------------

    def enroll(self, identity_name: str, images: list[bytes | IO[bytes] | str]) -> EnrollResult:
        """Register a person in the ArcFace index.

        ``images`` can be file paths (str), raw JPEG bytes, or file-like
        objects.  Each must contain exactly one face.
        """
        import io as _io
        files = []
        for i, src in enumerate(images):
            if isinstance(src, str):
                f = open(src, "rb")
                files.append(("images", (Path(src).name, f, "image/jpeg")))
            elif hasattr(src, "read"):
                files.append(("images", (f"img_{i}.jpg", src, "image/jpeg")))
            else:
                files.append(("images", (f"img_{i}.jpg",
                                         _io.BytesIO(src), "image/jpeg")))
        data = {"identity": identity_name}
        body = self._request("POST", "/enroll", data=data, files=files)
        for _, f in files:
            if hasattr(f[1], "close"):
                try:
                    if isinstance(f[1], tuple):
                        pass  # BytesIO
                    else:
                        f[1].close()
                except Exception:
                    pass
        return EnrollResult.from_dict(body)

    # ------------------------------------------------------------------
    # Natural-language Query
    # ------------------------------------------------------------------

    def ask(self, question: str) -> QueryResult:
        """Ask a natural-language question about recorded events.

        Arabic and English are both supported by the backend VLM.
        """
        body = self._request("POST", "/query", json={"question": question})
        return QueryResult.from_dict(body)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        url = self.base_url
        user = self._username or "token" if self._token else "anonymous"
        return f"<IObserve {url} [{user}]>"


# ===========================================================================
# Example / smoke test (run directly)
# ===========================================================================

if __name__ == "__main__":
    import sys
    # Accept one-shot credentials from CLI args:  python iobserve.py [user] [pass]
    cli_user = sys.argv[1] if len(sys.argv) > 1 else None
    cli_pass = sys.argv[2] if len(sys.argv) > 2 else None
    api = IObserve(username=cli_user, password=cli_pass)

    # 1 – Health (no auth)
    try:
        h = api.health()
        print(f"[OK] Health: {h}")
    except IObserveError as exc:
        print(f"[ERR] Server unreachable: {exc}")
        sys.exit(1)

    # 2 – Auth / whoami
    try:
        me = api.whoami()
        print(f"[AUTH] Authenticated as: {me.get('username', '?')}")
    except IObserveError:
        print("[AUTH] Not authenticated (expected when no token or user set)")

    # 3 – Counters
    try:
        c = api.counters()
        print(f"[DATA] Counters: {c.events} events, {c.face_detections} detections, "
              f"{c.identified} identified / {c.unknown} unknown, "
              f"{c.identities} identities")
        print(f"       Max similarity: {c.max_similarity:.1%}  "
              f"Latest action: {c.latest_action or '--'}")
    except IObserveError as exc:
        print(f"[WARN] Counters: {exc}")

    # 4 – System status
    try:
        s = api.system_status()
        pg = "OK" if s.postgres and s.postgres.ok else "FAIL"
        fi = "OK" if s.face_index and s.face_index.ok else "FAIL"
        fa = "OK" if s.face and s.face.ok else "FAIL"
        vlm = "OK" if s.vlm and s.vlm.ok else "SKIP"
        print(f"[STATUS] PG [{pg}] Index [{fi}] Face [{fa}] VLM [{vlm}] | "
              f"{len(s.identities)} identities, {s.counts.get('images',0)} images")
    except IObserveError as exc:
        print(f"[WARN] Status: {exc}")

    # 5 – Recent events
    try:
        ev = api.search_events(limit=3)
        if ev:
            print(f"[EVENTS] {len(ev)} recent:")
            for e in ev:
                print(f"  {e.recorded_at}  {e.identity or '--':>12}  {e.event_type}")
        else:
            print("[EVENTS] No events yet")
    except IObserveError as exc:
        print(f"[WARN] Events: {exc}")

    # 6 – Recordings
    try:
        recs = api.live_recordings()
        print(f"[RECS] {len(recs)} recording(s)")
    except IObserveError as exc:
        print(f"[WARN] Recordings: {exc}")

    # 7 – Enrolled identities from status
    try:
        s2 = api.system_status()
        if s2.identities:
            for id_ in s2.identities:
                print(f"[ID] {id_}")
    except IObserveError:
        pass

    print("\n[DONE] I-Observe integration kit verified and working.")
