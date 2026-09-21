"""Non-privileged world-state blackboard. Facts come only from observations."""
import copy
import threading

import numpy as np


class ObjectRecord:
    def __init__(self, id, visible, bbox, point, confidence, frame_id,
                 placement=None, top_z=None, bottom_z=None, extent=None):
        self.id = id
        self.visible = bool(visible)
        self.bbox = list(bbox)
        self.point = [float(v) for v in point]
        self.confidence = float(confidence)
        self.frame_id = int(frame_id)
        self.placement = placement
        self.top_z = float(top_z) if top_z is not None else self.point[2] + .007
        self.bottom_z = float(bottom_z) if bottom_z is not None else self.top_z - .036
        self.extent = list(extent) if extent is not None else [.024, .024, .036]

    def to_dict(self):
        return {"id": self.id, "visible": self.visible, "bbox": self.bbox,
                "point": list(self.point), "confidence": self.confidence,
                "frame_id": self.frame_id, "placement": self.placement,
                "top_z": self.top_z, "bottom_z": self.bottom_z,
                "extent": list(self.extent)}


class WorldState:
    def __init__(self, catalog=None):
        self.lock = threading.RLock()
        self.objects = {}
        self.gripper_open = True
        self.held_object = None
        self.grasp_target = None
        self.grasp_offset = None
        self.grasp_bottom_offset = None
        self.catalog = {s["id"]: copy.deepcopy(s) for s in (catalog or [
            {"id": "A", "kind": "block", "label": "红色物块 A"},
            {"id": "B", "kind": "block", "label": "蓝色物块 B"},
            {"id": "box", "kind": "box", "label": "绿色盒子"},
        ])}
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
            if open_:
                released = self.objects.get(self.held_object or self.grasp_target)
                if released is not None:
                    released.placement = None
                    released.visible = False
                self.held_object = None
                self.grasp_target = None
                self.grasp_offset = None
                self.grasp_bottom_offset = None

    def set_held(self, object_id):
        with self.lock:
            self.held_object = object_id

    def object_point(self, object_id):
        with self.lock:
            rec = self.objects.get(object_id)
            return np.array(rec.point) if rec is not None else None

    def mark_missing(self, target):
        with self.lock:
            if target in self.objects:
                self.objects[target].visible = False

    def is_box(self, target):
        return self.catalog.get(target, {}).get("kind") == "box"

    def _infer_placement(self, target_id, point):
        if self.held_object == target_id:
            return "gripper"
        # A point alone proves neither containment nor support contact.
        # VerifyState establishes relations from fresh measured geometry.
        return None

    def apply_observation(self, target, locate_result, frame_id):
        with self.lock:
            point = np.asarray(locate_result["point"], dtype=float)
            placement = self._infer_placement(target, point)
            rec = ObjectRecord(
                target, True, list(locate_result["bbox"]), point,
                float(locate_result.get("confidence", 0.9)), frame_id, placement,
                locate_result.get("top_z"), locate_result.get("bottom_z"),
                locate_result.get("extent"))
            self.objects[target] = rec
            self.frame_id = int(frame_id)
            return copy.copy(rec)

    def snapshot(self):
        with self.lock:
            return {
                "frame_id": self.frame_id,
                "gripper_open": self.gripper_open,
                "held_object": self.held_object,
                "grasp_target": self.grasp_target,
                "catalog": copy.deepcopy(self.catalog),
                "objects": {k: v.to_dict() for k, v in self.objects.items()},
            }
