"""Webcam-free check, safe to run on every push: uv run python scripts/smoke_test.py (exit 0 = ok)."""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "dashboard"))
import app

c = app.app.test_client()
assert c.get("/").status_code == 200
ok = dict(ear=.3, perclos=.05, mar=.25, pitch=0, yaw=0, phone=False)
bad = {**ok, "ear": .1, "perclos": .5}
assert app.analyse(ok)[2] == "ok" and app.analyse(bad)[2] == "danger"
assert c.post("/api/event", json=bad).json["ok"]
assert app.analyse({"face": False})[2] == "warn"
assert c.post("/api/event", json={"calibrating": 3.0}).json["ok"]
assert c.post("/api/recalibrate").json["ok"] and c.post("/api/event", json=ok).json["recalibrate"]
assert app.level >= 1, "escalation did not start"
assert c.post("/api/ack").json["ok"] and app.level == 0
print("test passed")
