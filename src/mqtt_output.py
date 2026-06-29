"""MQTT publisher for live people positions."""
import json
import os
import time
from collections import deque


class MqttPositionPublisher:
    def __init__(self, cfg):
        self.enabled = bool(cfg.get("enabled", False))
        if not self.enabled:
            return

        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:
            raise RuntimeError(
                "MQTT output is enabled but paho-mqtt is not installed. "
                "Run: pip install paho-mqtt"
            ) from exc

        self._mqtt = mqtt
        self.broker = cfg.get("broker", "localhost")
        self.port = int(cfg.get("port", 1883))
        self.username = os.environ.get("MQTT_USERNAME") or cfg.get("username")
        self.password = os.environ.get("MQTT_PASSWORD") or cfg.get("password")
        self.serial = str(cfg.get("serial", "people_tracker_3d"))
        self.topic = cfg.get("topic", f"smx/device/{self.serial}/position")
        self.qos = int(cfg.get("qos", 0))
        self.retain = bool(cfg.get("retain", False))
        self.publish_hz = float(cfg.get("publish_hz", 10.0))
        self.trail_length = int(cfg.get("trail_length", 10))

        self._last_publish = 0.0
        self._last_seen = {}
        self._trails = {}
        self._connected = False

        self.client = self._make_client()
        if self.username is not None:
            self.client.username_pw_set(self.username, self.password)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)

    def _make_client(self):
        client_id = f"people-tracker-3d-{os.getpid()}"
        if hasattr(self._mqtt, "CallbackAPIVersion"):
            return self._mqtt.Client(
                self._mqtt.CallbackAPIVersion.VERSION1,
                client_id=client_id,
            )
        return self._mqtt.Client(client_id=client_id)

    def _on_connect(self, _client, _userdata, _flags, rc):
        self._connected = rc == 0
        if self._connected:
            print(f"[mqtt] connected to {self.broker}:{self.port}, publishing {self.topic}")
        else:
            print(f"[mqtt] connect failed with rc={rc}")

    def _on_disconnect(self, _client, _userdata, rc):
        self._connected = False
        if rc != 0:
            print(f"[mqtt] disconnected unexpectedly rc={rc}; will reconnect")

    def start(self):
        if not self.enabled:
            return
        print(f"[mqtt] connecting to {self.broker}:{self.port} ...")
        self.client.connect_async(self.broker, self.port, keepalive=60)
        self.client.loop_start()

    def stop(self):
        if not self.enabled:
            return
        self.client.disconnect()
        self.client.loop_stop()

    def publish_positions(self, people):
        if not self.enabled or not self._connected:
            return

        now = time.time()
        if self.publish_hz > 0 and now - self._last_publish < 1.0 / self.publish_hz:
            return
        self._last_publish = now

        payload = self._build_payload(people, now)
        self.client.publish(
            self.topic,
            json.dumps(payload, separators=(",", ":")),
            qos=self.qos,
            retain=self.retain,
        )

    def _build_payload(self, people, now):
        object_list = {}
        active_ids = set()

        for person in people:
            world = person.get("world")
            if world is None:
                continue

            obj_id = str(person.get("gid", person.get("id")))
            active_ids.add(obj_id)

            x_m, y_m = float(world[0]), float(world[1])
            x_cm, y_cm = x_m * 100.0, y_m * 100.0
            speed = self._speed_mps(obj_id, x_m, y_m, now)
            trail = self._trail(obj_id, x_cm, y_cm)

            object_list[obj_id] = {
                "ID": obj_id,
                "X": [x for x, _y in trail],
                "Y": [y for _x, y in trail],
                "speed": speed,
                "object_type": "pedestrian",
            }

        for stale_id in set(self._trails) - active_ids:
            self._trails.pop(stale_id, None)
            self._last_seen.pop(stale_id, None)

        return {
            "messageType": "position",
            "collector_serial": self.serial,
            "total_detected_objects": len(object_list),
            "target_count": 0,
            "object_list": object_list,
        }

    def _speed_mps(self, obj_id, x_m, y_m, now):
        prev = self._last_seen.get(obj_id)
        self._last_seen[obj_id] = (x_m, y_m, now)
        if prev is None:
            return 0.0

        px, py, pt = prev
        dt = max(now - pt, 1e-6)
        return (((x_m - px) ** 2 + (y_m - py) ** 2) ** 0.5) / dt

    def _trail(self, obj_id, x_cm, y_cm):
        trail = self._trails.get(obj_id)
        if trail is None:
            trail = deque(maxlen=self.trail_length)
            self._trails[obj_id] = trail
        trail.appendleft((x_cm, y_cm))
        return list(trail)
