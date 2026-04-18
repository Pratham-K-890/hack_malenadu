"""
Orbital Foundry — FastAPI backend
Connects to the Node.js SSE server (localhost:3000), runs predictions,
and exposes REST + SSE endpoints consumed by the React dashboard.
"""
import os, sys, json, time, math, threading, asyncio, logging
from collections import deque
from datetime import datetime, timezone
from typing import AsyncGenerator, List

import numpy as np
import joblib
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("orbital")

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODE_URL = os.environ.get("NODE_URL", "http://localhost:3000")
MACHINE_IDS = ["CNC_01", "CNC_02", "PUMP_03", "CONVEYOR_04"]
SENSORS     = ["temperature_C", "vibration_mm_s", "rpm", "current_A"]
BUFFER_SIZE = 120

# ── Machine display config (names + images from Stitch design) ─────────────────
MACHINE_META = {
    "CNC_01": {
        "display_name": "CNC_01",
        "subtitle":     "Precision CNC Mill",
        "asset_id":     "M01",
        "location":     "Bay A-1",
        "image":        "https://lh3.googleusercontent.com/aida-public/AB6AXuCFtqDTDGO5SlTlTyuOyKQm3dZNWHxH5sw8hfgqGXObZfjX7PmpiSdFp4N99e5-dQObUw9JsSMuSCdhz1N5rpFLE8EobQ3aq8XXHT0TgJkPbq618LcTLbl996P2V8kCtVAJhUmVtVDuMarWDRxrWnHlLjm77RSuhOxRT4XYezACDK14wLUo7XuMYn4TDNzzSNH8vUMklre2s669pcbuaI2TrTyZdoSHuDFPxyFnTCK2z8t0i164F5TZ2Qbvb9UoVon7LoSCJzeTBw3P",
    },
    "CNC_02": {
        "display_name": "CNC_02",
        "subtitle":     "Kinetic Robotic Arm",
        "asset_id":     "M02",
        "location":     "Bay A-2",
        "image":        "https://lh3.googleusercontent.com/aida-public/AB6AXuBb1iOziMFbpqfi_wa1PRRNQbCj38zOmxNsviAi1p_wFTpz8DJOtIpvDny4Pr6pExGFOpVLXDLynem05wQp9ilYgbpDEqo9Y0P6P030xrDwOgYhbA_lq2sIVn3YqKBxoO87rhBbKXqqWSCcJw1qqkNj9M78lxZhISjj8PuvnKoKI1pn3Tlkl9k-a-X9IKcbZaeCW8peX4jnT5rWblChb0SwXUGbz3eyQUvoeWJgQI8M-puX2eK8hdOCiTIOBU4jotpxTSmSVr_qbBnY",
    },
    "PUMP_03": {
        "display_name": "PUMP_03",
        "subtitle":     "Hydraulic Pump",
        "asset_id":     "M03",
        "location":     "Bay B-1",
        "image":        "https://lh3.googleusercontent.com/aida-public/AB6AXuAO-E3K5Csx1DG790XnuvV3hoFCB9Cahya1uopkebZxnXc9nwEyfYZ8RLPtLPURLA_d2qFRkLn8nMshFcpu79PVA_8m34Yxp2o8DZ5jEqnt-X5DJxMZQm9tt9hc6HHEtP-QMHPizgBx0W0zDKyJ3KBNjBxcbn1DaJJfEnNBT7nIwxg2m0kjRl8YoP58011mnBVQaQTT3NNcn9_21Ie59NyTQPVbT7owYV-HQ8oLK-Zkpj4TGG6N_ut3xbwpNI2D1gDRCjwsRyF2qtmp",
    },
    "CONVEYOR_04": {
        "display_name": "CONVEYOR_04",
        "subtitle":     "Belt Conveyor System",
        "asset_id":     "M04",
        "location":     "Bay C-1",
        "image":        "https://lh3.googleusercontent.com/aida-public/AB6AXuCBjBdjbVz2eS06iJnR_qeKuX_jgb5RaKicPGl3In8ci8K9GXW7DfrI6T1vXE2VWe2WR5xSVIQrphNpbJYUqnvkSCSFcCK8UCKdCwPlI7yc1UAQkhsuW3h0kOwckw3xZJ4dbozgc6oD6-N8dkAjm3IKbOUD7wqtmmdwnaWMG9jAOMwdOGXFBqEqyeayZMv66Hotp9SzV6XhWO8uS2EbEIbJSSR3a3KcvbMSyIOG2Xu4JCPAiBuOBqmLMrSS1ETXx6ftmpfYFcogianT",
    },
}

# ── Thresholds ─────────────────────────────────────────────────────────────────
Z_DEFAULT = 3.0

# Z-score thresholds for the sigmoid risk formula (matches app.py _INITIAL_Z_THRESHOLD)
_Z_THRESHOLD = {
    "CNC_01":      2.7,   # lower — CNC_01 rarely faults, need early detection
    "CNC_02":      2.8,   # slightly sensitive — faults frequently
    "PUMP_03":     2.5,   # most sensitive — need to recover FNs
    "CONVEYOR_04": 3.3,   # higher guard — belt dynamics inflate Z on normal ops
}

# Risk_pct thresholds for status determination (matches app.py overrides)
_RISK_ALERT_THRESHOLD = {
    "CNC_01":      55.0,
    "CNC_02":      84.0,
    "CONVEYOR_04": 93.0,
    "PUMP_03":     60.0,
}

# Per-machine IF corroboration min Z (must match app.py _if_min_z_map)
_IF_MIN_Z = {"CONVEYOR_04": 5.0, "CNC_02": 3.5, "PUMP_03": 2.0}

_MAX_SLOPE = {"temperature_C": 0.5, "vibration_mm_s": 0.05, "rpm": 2.0, "current_A": 0.1}

# Min consecutive above-threshold ticks before status changes to warning/critical
_MIN_BREACH = {"CNC_02": 1, "CONVEYOR_04": 3, "CNC_01": 1, "PUMP_03": 1}

# ── In-memory state ────────────────────────────────────────────────────────────
_state:         dict  = {}
_buffers:       dict  = {mid: deque(maxlen=BUFFER_SIZE) for mid in MACHINE_IDS}
_breach_counts: dict  = {mid: 0 for mid in MACHINE_IDS}
_baselines:dict  = {}
_models:   dict  = {}
_alerts:   list  = []
_loop:     asyncio.AbstractEventLoop | None = None
_subscribers: List[asyncio.Queue] = []


# ── Asset loader ───────────────────────────────────────────────────────────────
def _load_assets():
    global _baselines, _models
    bl = os.path.join(ROOT, "baselines.json")
    if os.path.isfile(bl):
        with open(bl) as f:
            _baselines = json.load(f)
        log.info("Loaded baselines.json")
    else:
        log.warning("baselines.json not found — z-score detection disabled")

    for mid in MACHINE_IDS:
        path = os.path.join(ROOT, f"model_{mid}.pkl")
        if os.path.isfile(path):
            try:
                _models[mid] = joblib.load(path)
                log.info(f"Loaded model_{mid}.pkl")
            except Exception as e:
                log.warning(f"Could not load {path}: {e}")

    for mid in MACHINE_IDS:
        _state[mid] = {
            "id":             mid,
            "display_name":   MACHINE_META[mid]["display_name"],
            "subtitle":       MACHINE_META[mid]["subtitle"],
            "asset_id":       MACHINE_META[mid]["asset_id"],
            "location":       MACHINE_META[mid]["location"],
            "image":          MACHINE_META[mid]["image"],
            "status":         "healthy",
            "risk_pct":       0.0,
            "temperature_C":  60.0,
            "vibration_mm_s": 1.0,
            "rpm":            3000.0,
            "current_A":      8.0,
            "if_anomaly":     False,
            "polyfit_score":  0.0,
            "server_status":  "running",
            "connection":     "connecting",
            "updated_at":     datetime.now(timezone.utc).isoformat(),
        }


# ── Prediction helpers ─────────────────────────────────────────────────────────
def _sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def _predict(machine_id: str, latest: dict) -> tuple[float, bool, float]:
    if machine_id not in _baselines:
        return 0.0, False, 0.0

    b = _baselines[machine_id]
    z_scores = {}
    for s in SENSORS:
        val  = float(latest.get(s, b[s]["mean"]))
        mean = float(b[s]["mean"])
        std  = max(float(b[s]["std"]), abs(mean) * 0.03, 1e-3)
        z_scores[s] = abs((val - mean) / std)

    z_thresh = _Z_THRESHOLD.get(machine_id, Z_DEFAULT)
    max_z    = max(z_scores.values()) if z_scores else 0.0

    # IsolationForest
    if_anomaly = False
    if machine_id in _models:
        try:
            X = np.array([[
                float(latest.get("temperature_C",  0)),
                float(latest.get("vibration_mm_s", 0)),
                float(latest.get("rpm",            0)),
                float(latest.get("current_A",      0)),
            ]])
            if_anomaly = int(_models[machine_id].predict(X)[0]) == -1
        except Exception:
            pass

    # IF corroboration gate
    if_min_z = _IF_MIN_Z.get(machine_id, 2.5 if machine_id in _RISK_ALERT_THRESHOLD else 0.0)
    isolation_penalty = 0.3 if (if_anomaly and max_z >= if_min_z) else 0.0

    # Polyfit — per-machine slope overrides matching app.py
    _slope_override = {
        "CONVEYOR_04": {"temperature_C": 0.8, "vibration_mm_s": 0.22, "rpm": 6.0, "current_A": 0.35},
        "CNC_02":      {"temperature_C": 0.8, "vibration_mm_s": 0.10, "current_A": 0.18},
    }
    buf = list(_buffers[machine_id])
    polyfit_score = 0.0
    if len(buf) >= 30 and machine_id in _baselines:
        buf30 = buf[-30:]
        x_arr = np.arange(30, dtype=float)
        ms    = {**_MAX_SLOPE, **_slope_override.get(machine_id, {})}
        sevs  = []
        for s in SENSORS:
            vals = [float(r.get(s, b[s]["mean"])) for r in buf30]
            try:
                slope = abs(float(np.polyfit(x_arr, vals, 1)[0]))
                sevs.append(min(slope / ms[s] * 100.0, 100.0))
            except Exception:
                sevs.append(0.0)
        polyfit_score = max(sevs)

    trend_penalty = (polyfit_score / 100.0) * 0.2
    base_score    = (max_z - z_thresh) / max(z_thresh, 1e-6)
    raw           = base_score + isolation_penalty + trend_penalty
    risk_pct      = round(max(0.0, min(100.0, _sigmoid(raw) * 100.0)), 2)
    return risk_pct, if_anomaly, round(polyfit_score, 1)


def _status_from_risk(risk_pct: float, server_status: str, machine_id: str = "") -> str:
    is_fault    = server_status in ("warning", "fault")
    alert_thr   = _RISK_ALERT_THRESHOLD.get(machine_id, 65.0)
    crit_thr    = min(alert_thr + 12.0, 97.0)
    if risk_pct >= crit_thr or (is_fault and risk_pct >= alert_thr):
        return "critical"
    if risk_pct >= alert_thr or is_fault:
        return "warning"
    return "healthy"


# ── SSE broadcast (sync → async bridge) ───────────────────────────────────────
def _broadcast(payload: dict):
    if _loop is None:
        return
    dead = []
    for q in list(_subscribers):
        try:
            _loop.call_soon_threadsafe(q.put_nowait, payload)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            _subscribers.remove(q)
        except Exception:
            pass


# ── Per-machine SSE worker (background thread) ─────────────────────────────────
def _stream_worker(machine_id: str):
    import sseclient, requests as _req
    backoff = 1
    while True:
        try:
            resp = _req.get(
                f"{NODE_URL}/stream/{machine_id}",
                headers={"Accept": "text/event-stream"},
                stream=True,
                timeout=None,
            )
            client  = sseclient.SSEClient(resp)
            backoff = 1
            _state[machine_id]["connection"] = "live"
            log.info(f"[{machine_id}] SSE connected")

            for event in client.events():
                if not (event.data and event.data.strip()):
                    continue
                try:
                    data = json.loads(event.data)
                    data.setdefault("machine_id", machine_id)
                    data.setdefault("status",     "running")
                    _buffers[machine_id].append(data)

                    risk_pct, if_anomaly, pf_score = _predict(machine_id, data)

                    # Breach counter — require N consecutive above-threshold ticks
                    alert_thr = _RISK_ALERT_THRESHOLD.get(machine_id, 65.0)
                    if risk_pct >= alert_thr:
                        _breach_counts[machine_id] += 1
                    else:
                        _breach_counts[machine_id] = 0
                    breach_ok = _breach_counts[machine_id] >= _MIN_BREACH.get(machine_id, 1)

                    # Only promote to warning/critical if breach count satisfied
                    raw_status = _status_from_risk(risk_pct, data.get("status", "running"), machine_id)
                    status = raw_status if (breach_ok or raw_status == "healthy") else "healthy"

                    _state[machine_id].update({
                        "temperature_C":  round(float(data.get("temperature_C",  _state[machine_id]["temperature_C"])),  2),
                        "vibration_mm_s": round(float(data.get("vibration_mm_s", _state[machine_id]["vibration_mm_s"])), 2),
                        "rpm":            round(float(data.get("rpm",            _state[machine_id]["rpm"])),            0),
                        "current_A":      round(float(data.get("current_A",      _state[machine_id]["current_A"])),      2),
                        "status":         status,
                        "risk_pct":       risk_pct,
                        "if_anomaly":     if_anomaly,
                        "polyfit_score":  pf_score,
                        "server_status":  data.get("status", "running"),
                        "updated_at":     datetime.now(timezone.utc).isoformat(),
                    })
                    _broadcast({"type": "machine_update", "data": dict(_state[machine_id])})
                except Exception:
                    pass

        except Exception as e:
            _state[machine_id]["connection"] = "disconnected"
            log.warning(f"[{machine_id}] stream error: {e}. Retry in {backoff}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Orbital Foundry API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    global _loop
    _loop = asyncio.get_event_loop()
    _load_assets()
    for mid in MACHINE_IDS:
        t = threading.Thread(target=_stream_worker, args=(mid,), daemon=True)
        t.start()
    log.info("Orbital Foundry API started on http://0.0.0.0:8000")


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/api/machines")
def get_machines():
    return list(_state.values())


@app.get("/api/machines/{machine_id}")
def get_machine(machine_id: str):
    return _state.get(machine_id, {"error": "not found"})


@app.get("/api/machines/{machine_id}/history")
def get_history(machine_id: str, n: int = 30):
    buf = list(_buffers.get(machine_id, []))[-n:]
    return {
        "machine_id":  machine_id,
        "vibration":   [{"t": i, "v": float(r.get("vibration_mm_s", 0))} for i, r in enumerate(buf)],
        "temperature": [{"t": i, "v": float(r.get("temperature_C",  0))} for i, r in enumerate(buf)],
        "rpm":         [{"t": i, "v": float(r.get("rpm",            0))} for i, r in enumerate(buf)],
        "current":     [{"t": i, "v": float(r.get("current_A",      0))} for i, r in enumerate(buf)],
    }


@app.get("/api/stream")
async def sse_stream(request: Request):
    queue: asyncio.Queue = asyncio.Queue(maxsize=200)
    _subscribers.append(queue)

    async def generator() -> AsyncGenerator[str, None]:
        for mid in MACHINE_IDS:
            yield f"data: {json.dumps({'type': 'machine_update', 'data': dict(_state.get(mid, {}))})}\n\n"
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=2.0)
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
            except Exception:
                break

    async def cleanup():
        try:
            _subscribers.remove(queue)
        except Exception:
            pass

    request.state.cleanup = cleanup
    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


@app.post("/api/machines/{machine_id}/confirm")
def confirm(machine_id: str):
    _alerts.append({"machine_id": machine_id, "action": "confirmed", "ts": datetime.now(timezone.utc).isoformat()})
    return {"status": "ok"}


@app.post("/api/machines/{machine_id}/dismiss")
def dismiss(machine_id: str):
    _alerts.append({"machine_id": machine_id, "action": "dismissed", "ts": datetime.now(timezone.utc).isoformat()})
    return {"status": "ok"}


@app.get("/api/alerts")
def get_alerts():
    return _alerts[-20:]


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
