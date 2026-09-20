"""Non-privileged world-state blackboard. Facts come only from observations."""
import copy
import threading

import numpy as np


class ObjectRecord:
    def __init__(self, id, visible, bbox, point, confidence, frame_id,
                 placement=None):
        self.id = id
        self.visible = bool(visible)
        self.bbox = list(bbox)
        self.point = [float(v) for v in point]
        self.confidence = float(confidence)
        self.frame_id = int(frame_id)
        self.placement = placement

    def to_dict(self):
        return {"id": self.id, "visible": self.visible, "bbox": self.bbox,
                "point": list(self.point), "confidence": self.confidence,
                "frame_id": self.frame_id, "placement": self.placement}


class WorldState:
    def __init__(self):
        self.lock = threading.RLock()
        self.objects = {}
        self.gripper_open = True
        self.held_object = None
        self.qpos = None
        self.frame_id = 0
        self._tick = 0

    def tick(self):
        with self.lock:
            self._tick += 1
            self.frame_id = self._tick
            return self._tick

    def update_object(self, record):
        with self.lock:
            self.objects[record.id] = record

    def set_gripper(self, open_):
        with self.lock:
            self.gripper_open = bool(open_)

    def set_held(self, object_id):
        with self.lock:
            self.held_object = object_id

    def object_point(self, object_id):
        with self.lock:
            rec = self.objects.get(object_id)
            return np.array(rec.point) if rec is not None else None

    def _infer_placement(self, target_id, point):
        if target_id == "box":
            return "table"
        if self.held_object == target_id:
            return "gripper"
        box = self.objects.get("box")
        if box is not None:
            xy = np.hypot(point[0] - box.point[0], point[1] - box.point[1])
            if xy < 0.045 and box.point[2] - 0.01 <= point[2] <= box.point[2] + 0.07:
                return "box"
        return "table"

    def apply_observation(self, target, locate_result, frame_id):
        with self.lock:
            point = np.asarray(locate_result["point"], dtype=float)
            placement = self._infer_placement(target, point)
            rec = ObjectRecord(
                target, True, list(locate_result["bbox"]), point,
                float(locate_result.get("confidence", 0.9)), frame_id, placement)
            self.objects[target] = rec
            self.frame_id = int(frame_id)
            return copy.copy(rec)

    def snapshot(self):
        with self.lock:
            return {
                "frame_id": self.frame_id,
                "gripper_open": self.gripper_open,
                "held_object": self.held_object,
                "objects": {k: v.to_dict() for k, v in self.objects.items()},
            }
