from __future__ import annotations

import json
import math
import os
import re
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class Severity(str, Enum):
    HARD = "hard"
    SOFT = "soft"
    INFO = "info"


@dataclass
class Violation:
    rule: str
    severity: Severity
    object_id: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"[{self.severity.value.upper():4s}] {self.rule}: {self.message}"

@dataclass
class Success:
    rule: str
    object_id: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return f"[PASS] {self.rule}: {self.message}"

@dataclass
class CheckResult:
    rule_name: str
    n_checked: int
    n_passed: int
    violations: List[Violation] = field(default_factory=list)
    successes: List[Success] = field(default_factory=list)

    @property
    def score(self) -> float:
        if self.n_checked == 0:
            return 1.0
        return self.n_passed / self.n_checked


@dataclass
class VerificationResult:
    rule_results: Dict[str, CheckResult]
    violations: List[Violation]
    successes: List[Success] = field(default_factory=list)

    @property
    def sp_score(self) -> float:
        if not self.rule_results:
            return 1.0
        scores = [r.score for r in self.rule_results.values()]
        return sum(scores) / len(scores)

    @property
    def sp_score_weighted(self) -> float:
        weights = {
            "scale": 0.25,
            "support": 0.0,
            "room_zone": 0.0,
            "containment": 0.0,
            "overlap": 0.0,
            "overlap_obb": 0.0,
            "orientation": 0.25,
            "cooccurrence": 0.25,
            "completeness": 0.25,
            "accessibility": 0.0,
        }
        total_w = 0.0
        total_s = 0.0
        for name, result in self.rule_results.items():
            w = weights.get(name, 0.10)
            total_w += w
            total_s += w * result.score
        return total_s / total_w if total_w > 0 else 1.0

    @property
    def by_rule(self) -> Dict[str, float]:
        return {name: r.score for name, r in self.rule_results.items()}

    @property
    def n_hard_violations(self) -> int:
        return sum(1 for v in self.violations if v.severity == Severity.HARD)

    def summary(self) -> str:
        lines = [
            f"Semantic Plausibility Score: {self.sp_score:.3f}  "
            f"(weighted: {self.sp_score_weighted:.3f})",
            f"Hard violations: {self.n_hard_violations}  |  "
            f"Total violations: {len(self.violations)}",
            "",
            "Per-rule breakdown:",
        ]
        for name, result in self.rule_results.items():
            status = "✓" if result.score == 1.0 else "✗"
            lines.append(
                f"  {status} {name:22s}  "
                f"{result.n_passed}/{result.n_checked}  "
                f"({result.score:.1%})"
            )
        if self.violations:
            lines.append("")
            lines.append("Violations:")
            for v in self.violations:
                lines.append(f"  {v}")
        if self.successes:
            lines.append("")
            lines.append("Successes:")
            for s in self.successes[:200]:
                lines.append(f"  {s}")
            if len(self.successes) > 200:
                lines.append(f"  ... ({len(self.successes)-200} more)")
        return "\n".join(lines)


@dataclass
class SceneObject:
    obj_id: str
    category: str
    x: float
    y: float
    z: float
    width: float
    height: float
    depth: float
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    is_2d: bool = False
    _source: str = "format3"

    @property
    def z_min(self) -> float:
        return self.z - self.height / 2.0

    @property
    def z_max(self) -> float:
        return self.z + self.height / 2.0

    @property
    def bbox_2d(self) -> Tuple[float, float, float, float]:
        hw = self.width / 2.0
        hd = self.depth / 2.0
        return (self.x - hw, self.y - hd, self.x + hw, self.y + hd)


@dataclass
class RoomInfo:
    room_type: str
    bounds: Optional[Tuple[float, float, float, float]] = None
    doors: List[Dict[str, float]] = field(default_factory=list)
    windows: List[Dict[str, float]] = field(default_factory=list)


CATEGORY_ALIASES: Dict[str, str] = {
    "chair": "chair", "dining chair": "chair", "desk chair": "chair",
    "office chair": "office_chair", "swivel chair": "office_chair",
    "armchair": "armchair", "lounge chair": "armchair", "accent chair": "armchair",
    "sofa": "sofa", "couch": "sofa", "loveseat": "sofa", "settee": "sofa",
    "sectional": "sofa", "l-shaped sofa": "sofa",
    "stool": "stool", "bar stool": "stool", "counter stool": "stool",
    "bench": "bench", "ottoman": "ottoman", "pouf": "ottoman", "bar counter": "counter",
    "recliner": "recliner", "rocking chair": "rocking_chair", "office chair": "chair",
    "bean bag": "bean_bag", "beanbag": "bean_bag", "bean bag chair": "bean_bag", "lab table": "table", "conference table": "table",
    "table": "table", "desk": "desk", "writing desk": "desk",
    "computer desk": "desk", "study desk": "desk",
    "dining table": "dining_table", "kitchen table": "dining_table",
    "coffee table": "coffee_table", "cocktail table": "coffee_table",
    "side table": "side_table", "end table": "side_table", "accent table": "side_table",
    "nightstand": "nightstand", "bedside_table": "table", "bedside table": "table", "night stand": "nightstand",
    "console table": "console_table", "console": "console_table",
    "dressing table": "dressing_table",
    "bed": "bed", "single bed": "bed", "double bed": "bed", "double_bed": "bed",
    "king bed": "bed", "queen bed": "bed", "bunk bed": "bunk_bed",
    "crib": "crib", "baby bed": "crib", "children bed": "bed",
    "wardrobe": "wardrobe", "closet": "wardrobe", "armoire": "wardrobe",
    "bookshelf": "bookshelf", "bookcase": "bookshelf", "shelving": "bookshelf",
    "shelf": "shelf", "wall shelf": "wall_shelf", "floating shelf": "wall_shelf",
    "cabinet": "cabinet", "cupboard": "cabinet", "storage cabinet": "cabinet",
    "dresser": "dresser", "chest of drawers": "dresser", "drawer": "dresser",
    "tv stand": "tv_stand", "media console": "tv_stand", "entertainment center": "tv_stand",
    "tv_stand": "tv_stand",
    "TV_stand": "tv_stand",
    "TV stand": "tv_stand",
    "sideboard": "sideboard", "buffet": "sideboard", "credenza": "sideboard",
    "filing cabinet": "filing_cabinet", "file cabinet": "filing_cabinet",
    "cabinet/shelf/desk": "cabinet",
    "pier/stool": "stool",
    "king-size bed": "bed",
    "kids bed": "bed",
    "single bed": "bed",
    "bunk bed": "bunk_bed",
    "bed frame": "bed",
    "round end table": "side_table",
    "dining chair": "chair",
    "lounge chair/cafe chair/office chair": "chair",
    "dressing chair": "chair",
    "classic chinese chair": "chair",
    "barstool": "stool",
    "armchair": "armchair",
    "three-seat/multi-seat sofa": "sofa",
    "two-seat sofa": "sofa",
    "l-shaped sofa": "sofa",
    "lazy sofa": "sofa",
    "chaise longue sofa": "sofa",
    "corner/side table": "side_table",
    "sideboard/side cabinet/console table": "sideboard",
    "children cabinet": "cabinet",
    "drawer chest/corner cabinet": "dresser",
    "shelf": "shelf",
    "wine cabinet": "cabinet",
    "tv stand": "tv_stand",
    "wardrobe": "wardrobe",
    "bookcase/jewelry armoire": "bookshelf",
    "ceiling lamp": "ceiling_lamp",
    "pendant lamp": "pendant_lamp",
    "floor lamp": "floor_lamp",
    "wall lamp": "wall_lamp",
    "table lamp": "table_lamp",
    "bulb": "lamp",
    "desk": "desk",
    "coffee table": "coffee_table",
    "dining table": "dining_table",
    "dressing table": "dressing_table",
    "nightstand": "nightstand",
    "lamp": "lamp", "table lamp": "table_lamp", "desk lamp": "table_lamp",
    "floor lamp": "floor_lamp", "standing lamp": "floor_lamp",
    "ceiling lamp": "ceiling_lamp", "chandelier": "chandelier",
    "pendant lamp": "pendant_lamp", "pendant light": "pendant_lamp",
    "wall lamp": "wall_lamp", "sconce": "wall_lamp", "potted plant": "plant", "magazine rack": "shelf",
    "tv": "tv", "television": "tv", "monitor": "monitor", "computer monitor": "monitor", "study_table": "table",
    "laptop": "laptop", "computer": "computer", "desktop computer": "computer",
    "speaker": "speaker", "soundbar": "speaker",
    "printer": "printer",
    "refrigerator": "refrigerator", "fridge": "refrigerator",
    "oven": "oven", "stove": "stove", "wood-burning stove": "stove", "cooktop": "stove",
    "microwave": "microwave", "dishwasher": "dishwasher", "bread display shelf": "shelf",
    "washing machine": "washing_machine", "washer": "washing_machine",
    "dryer": "dryer", "air conditioner": "air_conditioner", "grand piano": "piano",
    "toilet": "toilet", "sink": "sink", "bathroom sink": "sink",
    "bathtub": "bathtub", "tub": "bathtub", "shower": "shower",
    "mirror": "mirror", "bathroom mirror": "mirror",
    "rug": "rug", "carpet": "rug", "area rug": "rug",
    "curtain": "curtain", "drape": "curtain", "window curtain": "curtain",
    "painting": "painting", "picture": "painting", "wall art": "painting",
    "photo frame": "photo_frame", "picture frame": "photo_frame",
    "vase": "vase", "flower vase": "vase",
    "plant": "plant", "houseplant": "plant",
    "clock": "clock", "wall clock": "clock",
    "pillow": "pillow", "cushion": "pillow", "throw pillow": "pillow",
    "blanket": "blanket", "throw": "blanket",
    "umbrella": "patio_umbrella", "patio umbrella": "patio_umbrella",
    "grill": "grill", "bbq": "grill", "barbecue": "grill",
    "planter": "planter", "flower pot": "planter",
    "outdoor sofa": "outdoor_sofa", "garden bench": "garden_bench",
    "fire pit": "fire_pit", "firepit": "fire_pit",
    "lounge chair": "lounge_chair", "sun lounger": "lounge_chair", "chaise lounge": "lounge_chair",
    "door": "door", "window": "window",
    "ceiling fan": "ceiling_fan", "fan": "fan",
    "fireplace": "fireplace",
    "piano": "piano", "keyboard": "keyboard",
    "bread basket": "bread_basket",
    "basket": "basket",
}

def normalize_category(label: str) -> str:
    if not label:
        return "unknown"

    stripped = re.sub(r'[-_](\d+)$', '', label.strip())
    stripped = stripped.lower().strip()

    if stripped in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[stripped]

    spaced = stripped.replace("_", " ")
    if spaced in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[spaced]

    _ADJECTIVES = {
        "wooden", "wood", "metal", "glass", "plastic", "leather", "fabric",
        "large", "small", "big", "tiny", "tall", "short", "wide", "narrow",
        "modern", "classic", "vintage", "antique", "rustic", "industrial",
        "round", "square", "rectangular", "oval", "circular",
        "white", "black", "brown", "gray", "grey", "red", "blue", "green", "unit", "regular",
    }
    tokens = spaced.split()
    core_tokens = [t for t in tokens if t not in _ADJECTIVES]
    if core_tokens and core_tokens != tokens:
        core = " ".join(core_tokens)
        if core in CATEGORY_ALIASES:
            return CATEGORY_ALIASES[core]

    candidate = stripped.replace(" ", "_")
    return candidate


def normalize_room_type(name: str) -> str:
    if not name:
        return "bedroom"
    name = name.strip().replace(" ", "")
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    return name.lower()


def _build_reverse_alias_map() -> Dict[str, Set[str]]:
    reverse: Dict[str, Set[str]] = {}
    for alias, canonical in CATEGORY_ALIASES.items():
        reverse.setdefault(canonical, set()).add(alias)
        reverse.setdefault(canonical, set()).add(canonical)
    return reverse


_REVERSE_ALIASES = _build_reverse_alias_map()


def get_all_aliases(canonical: str) -> Set[str]:
    result = _REVERSE_ALIASES.get(canonical, set())
    result = set(result)
    result.add(canonical)
    result.add(canonical.replace("_", " "))
    result.add(canonical.replace(" ", "_"))
    return result


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


import numpy as np


def build_category_mesh_dims_from_json(mesh_dims_path: str) -> Dict[str, Tuple[float, float, float]]:
    with open(mesh_dims_path, "r") as f:
        raw = json.load(f)

    out: Dict[str, Tuple[float, float, float]] = {}
    for mesh_name, dims in raw.items():
        cat = normalize_category(mesh_name)
        if cat and cat != "unknown" and len(dims) == 3:
            w, d, h = float(dims[0]), float(dims[1]), float(dims[2])
            if w > 0 and d > 0 and h > 0:
                out[cat] = (w, d, h)
    return out


def blender_like_infer_height(target_w: float, target_d: float,
                              mesh_w: float, mesh_d: float, mesh_h: float) -> float:
    if mesh_w <= 0 or mesh_d <= 0 or mesh_h <= 0:
        return 1.0
    sx = target_w / mesh_w
    sy = target_d / mesh_d
    sz = (sx + sy) / 2.0
    h = mesh_h * sz
    return max(h, 0.01)


SKIP_LABELS = {"ego_person", "ego_agent", "person", "agent"}


def is_2d_scene(scene: dict) -> bool:
    return "Environment Name" in scene or "Environmental_note" in scene


def is_layoutvlm_scene(scene: dict) -> bool:
    if not isinstance(scene, dict):
        return False
    if "objects" in scene or "Environment Name" in scene or "room" in scene:
        return False
    for key, val in scene.items():
        if isinstance(val, dict) and "position" in val and "rotation" in val:
            return True
    return False


def convert_2d_scene(
    scene: dict,
    ontology: Optional[Dict[str, "OntologyEntry"]] = None,
    task_config: Optional[dict] = None,
    category_mesh_dims: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> dict:
    room_name = scene.get("Environment Name", "")
    if not room_name:
        note = scene.get("Environmental_note", "") or ""
        for candidate in ["bedroom", "living_room", "kitchen", "bathroom",
                          "dining_room", "office", "library", "hallway"]:
            if candidate.replace("_", " ") in note.lower():
                room_name = candidate
                break
        if not room_name:
            room_name = "bedroom"

    room_type = normalize_room_type(room_name)

    rx, ry = 5.0, 5.0
    if task_config:
        rd = task_config.get("room_dimension", {})
        rx = float(rd.get("room_dimension_x", rx))
        ry = float(rd.get("room_dimension_y", ry))

    converted = {
        "room": {
            "type": room_type,
            "bounds": {
                "x_min": -rx / 2.0, "y_min": 0.0,
                "x_max": rx / 2.0, "y_max": ry,
            },
            "doors": [],
            "windows": [],
        },
        "objects": [],
    }

    for obj in scene.get("objects", []):
        label = obj.get("label", "")
        label_lower = label.lower().strip()
        if label_lower in SKIP_LABELS or "ego" in label_lower:
            continue

        category = normalize_category(label)
        if category == "unknown":
            continue

        bbox = obj.get("bbox", [0, 0, 1, 1])
        pos = obj.get("position", [0, 0])

        width_2d = float(bbox[2]) if len(bbox) > 2 else 1.0
        depth_2d = float(bbox[3]) if len(bbox) > 3 else 1.0

        inferred_height = 1.0
        if category_mesh_dims and category in category_mesh_dims:
            mw, md, mh = category_mesh_dims[category]
            inferred_height = blender_like_infer_height(width_2d, depth_2d, mw, md, mh)
        elif ontology and category in ontology:
            entry = ontology[category]
            if entry.dim_height_min is not None and entry.dim_height_max is not None:
                inferred_height = (entry.dim_height_min + entry.dim_height_max) / 2.0

        rotation = obj.get("rotation_angle", None)
        if rotation is None:
            rotation = 0.0

        converted_obj = {
            "id": label,
            "category": category,
            "position": {
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": inferred_height / 2.0,
            },
            "dimensions": {
                "width": width_2d,
                "height": inferred_height,
                "depth": depth_2d,
            },
            "orientation": {
                "yaw": float(rotation),
                "pitch": 0.0,
                "roll": 0.0,
            },
            "_is_2d": True,
            "_source": "format3",
        }
        converted["objects"].append(converted_obj)

    return converted


def _estimate_dimensions_from_ontology(
    category: str,
    ontology: Optional[Dict[str, "OntologyEntry"]],
) -> Tuple[float, float, float]:
    if ontology and category in ontology:
        entry = ontology[category]
        w = entry.dim_width_median if entry.dim_width_median else 0.5
        d = entry.dim_depth_median if entry.dim_depth_median else 0.5
        h = entry.dim_height_median if entry.dim_height_median else 0.5
        return (w, d, h)

    DEFAULTS = {
        "chair": (0.5, 0.5, 0.9),
        "table": (1.2, 0.8, 0.75),
        "dining_table": (1.5, 0.9, 0.75),
        "desk": (1.2, 0.6, 0.75),
        "bed": (2.0, 1.5, 0.6),
        "sofa": (2.0, 0.9, 0.85),
        "shelf": (0.8, 0.3, 1.8),
        "cabinet": (0.8, 0.4, 1.0),
        "nightstand": (0.5, 0.4, 0.55),
        "tv": (1.0, 0.1, 0.6),
        "dishwasher": (0.6, 0.6, 0.85),
        "refrigerator": (0.7, 0.7, 1.7),
        "oven": (0.6, 0.6, 0.9),
    }
    return DEFAULTS.get(category, (0.5, 0.5, 0.5))


def convert_layoutvlm_scene(
    scene: dict,
    task_config: Optional[dict] = None,
    category_mesh_dims: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ontology: Optional[Dict[str, "OntologyEntry"]] = None,
) -> dict:
    room_type = "unknown"
    rx_min, ry_min = 0.0, 0.0
    rx_max, ry_max = 5.0, 5.0

    if task_config:
        desc = task_config.get("task_description", "")
        for candidate in ["bedroom", "living_room", "living room", "kitchen",
                          "bathroom", "dining_room", "dining room", "office",
                          "library", "hallway", "restaurant", "cafe",
                          "laundry_room", "laundry room"]:
            if candidate.replace("_", " ") in desc.lower():
                room_type = candidate.replace(" ", "_")
                break

        boundary = task_config.get("boundary", {})
        vertices = boundary.get("floor_vertices", [])
        if vertices:
            xs = [v[0] for v in vertices]
            ys = [v[1] for v in vertices]
            rx_min = min(xs)
            ry_min = min(ys)
            rx_max = max(xs)
            ry_max = max(ys)

        if not vertices:
            rd = task_config.get("room_dimension", {})
            rx = float(rd.get("room_dimension_x", 5))
            ry = float(rd.get("room_dimension_y", 5))
            rx_min, ry_min = -rx / 2.0, 0.0
            rx_max, ry_max = rx / 2.0, ry

    converted = {
        "room": {
            "type": room_type,
            "bounds": {
                "x_min": rx_min, "y_min": ry_min,
                "x_max": rx_max, "y_max": ry_max,
            },
            "doors": [],
            "windows": [],
        },
        "objects": [],
    }

    for obj_key, obj_data in scene.items():
        if not isinstance(obj_data, dict) or "position" not in obj_data:
            continue

        label_lower = obj_key.lower().strip()
        if label_lower in SKIP_LABELS or "ego" in label_lower:
            continue

        category = normalize_category(obj_key)
        if category == "unknown":
            continue

        pos = obj_data.get("position", [0, 0, 0])
        rot = obj_data.get("rotation", [0, 0, 0])

        px = float(pos[0]) if len(pos) > 0 else 0.0
        py = float(pos[1]) if len(pos) > 1 else 0.0
        pz = float(pos[2]) if len(pos) > 2 else 0.0

        rx = float(rot[0]) if len(rot) > 0 else 0.0
        ry = float(rot[1]) if len(rot) > 1 else 0.0
        rz = float(rot[2]) if len(rot) > 2 else 0.0

        if category_mesh_dims and category in category_mesh_dims:
            mw, md, mh = category_mesh_dims[category]
            width, depth, height = mw, md, mh
        else:
            width, depth, height = _estimate_dimensions_from_ontology(
                category, ontology
            )

        converted_obj = {
            "id": obj_key,
            "category": category,
            "position": {
                "x": px,
                "y": py,
                "z": pz,
            },
            "dimensions": {
                "width": width,
                "height": height,
                "depth": depth,
            },
            "orientation": {
                "yaw": rz,
                "pitch": rx,
                "roll": ry,
            },
            "_is_2d": False,
            "_source": "layoutvlm",
        }
        converted["objects"].append(converted_obj)

    return converted


def is_layoutvlm_task_config(task_config: dict) -> bool:
    if not task_config:
        return False
    boundary = task_config.get("boundary", {})
    if "floor_vertices" in boundary:
        return True
    assets = task_config.get("assets", {})
    for key in assets:
        base = key.rsplit("-", 1)[0]
        if len(base) == 32 and all(c in "0123456789abcdef" for c in base):
            return True
    return False


def build_expected_assets_from_layoutvlm_scene(
    scene_json: dict,
) -> Dict[str, int]:
    if not isinstance(scene_json, dict):
        return {}

    cat_counts: Dict[str, int] = {}
    for obj_key, obj_data in scene_json.items():
        if not isinstance(obj_data, dict) or "position" not in obj_data:
            continue
        label_lower = obj_key.lower().strip()
        if label_lower in SKIP_LABELS or "ego" in label_lower:
            continue
        canonical = normalize_category(obj_key)
        if canonical and canonical != "unknown":
            cat_counts[canonical] = cat_counts.get(canonical, 0) + 1

    return cat_counts


def build_config_from_layoutvlm_task(
    task_config: dict,
    category_mesh_dims: Optional[Dict[str, Tuple[float, float, float]]] = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    desc = task_config.get("task_description", "")
    for candidate in ["bedroom", "living_room", "living room", "kitchen",
                      "bathroom", "dining_room", "dining room", "office",
                      "library", "hallway", "restaurant", "cafe",
                      "laundry_room", "laundry room"]:
        if candidate in desc.lower():
            config["domain"] = candidate.replace(" ", "_")
            break
    if "domain" not in config:
        config["domain"] = "unknown"

    boundary = task_config.get("boundary", {})
    vertices = boundary.get("floor_vertices", [])
    if vertices:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        config["room_dimension_x"] = max(xs) - min(xs)
        config["room_dimension_y"] = max(ys) - min(ys)

    return config


def parse_scene(scene_json: Dict[str, Any]) -> Tuple[List[SceneObject], RoomInfo]:
    objects = []
    for obj in scene_json.get("objects", []):
        pos = obj.get("position", {})
        dims = obj.get("dimensions", {})
        orient = obj.get("orientation", {})
        objects.append(SceneObject(
            obj_id=obj.get("id", "unknown"),
            category=obj.get("category", "unknown"),
            x=float(pos.get("x", 0)),
            y=float(pos.get("y", 0)),
            z=float(pos.get("z", 0)),
            width=float(dims.get("width", 0)),
            height=float(dims.get("height", 0)),
            depth=float(dims.get("depth", 0)),
            yaw=float(orient.get("yaw", 0)),
            pitch=float(orient.get("pitch", 0)),
            roll=float(orient.get("roll", 0)),
            is_2d=obj.get("_is_2d", False),
            _source=obj.get("_source", "format3"),
        ))

    room_data = scene_json.get("room", {})
    bounds_data = room_data.get("bounds", {})
    bounds = None
    if bounds_data:
        bounds = (
            bounds_data.get("x_min", -5), bounds_data.get("y_min", -5),
            bounds_data.get("x_max", 5), bounds_data.get("y_max", 5),
        )

    room = RoomInfo(
        room_type=room_data.get("type", "unknown"),
        bounds=bounds,
        doors=room_data.get("doors", []),
        windows=room_data.get("windows", []),
    )

    return objects, room


@dataclass
class OrientationRelationships:
    back_to_wall_fraction: float = 0.0
    back_to_wall_mean_angle: float = 90.0
    back_to_wall_n_samples: int = 0
    faces_center_fraction: float = 0.0
    faces_center_mean_angle: float = 90.0
    faces_center_n_samples: int = 0
    faces_pair: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class OntologyEntry:
    category: str
    total_scene_count: int = 0
    dim_width_min: Optional[float] = None
    dim_width_max: Optional[float] = None
    dim_height_min: Optional[float] = None
    dim_height_max: Optional[float] = None
    dim_depth_min: Optional[float] = None
    dim_depth_max: Optional[float] = None
    dim_width_median: Optional[float] = None
    dim_height_median: Optional[float] = None
    dim_depth_median: Optional[float] = None
    room_fractions: Dict[str, float] = field(default_factory=dict)
    support_fractions: Dict[str, float] = field(default_factory=dict)
    z_center_p5: Optional[float] = None
    z_center_p95: Optional[float] = None
    z_bottom_p5: Optional[float] = None
    z_bottom_p95: Optional[float] = None
    cooccurrence: Dict[str, Any] = field(default_factory=dict)
    orientation_rel: OrientationRelationships = field(default_factory=OrientationRelationships)


def load_ontology(path: str) -> Dict[str, OntologyEntry]:
    with open(path, "r") as f:
        raw = json.load(f)

    ontology: Dict[str, OntologyEntry] = {}
    for cat, entry in raw.items():
        e = OntologyEntry(category=cat)
        e.total_scene_count = entry.get("count", 0)

        dims = entry.get("dimensions", {})
        if dims:
            w = dims.get("width_m", {}) or {}
            h = dims.get("height_m") or {}
            d = dims.get("depth_m") or {}
            e.dim_width_min = w.get("p5")
            e.dim_width_max = w.get("p95")
            e.dim_width_median = w.get("median")
            e.dim_height_min = h.get("p5")
            e.dim_height_max = h.get("p95")
            e.dim_height_median = h.get("median")
            e.dim_depth_min = d.get("p5")
            e.dim_depth_max = d.get("p95")
            e.dim_depth_median = d.get("median")

        for room, v in entry.get("room_associations", {}).items():
            if isinstance(v, dict):
                e.room_fractions[room] = v.get("fraction", 0.0)
            else:
                e.room_fractions[room] = float(v)

        for surf, v in entry.get("support_surfaces", {}).items():
            if isinstance(v, dict):
                e.support_fractions[surf] = v.get("fraction", 0.0)
            else:
                e.support_fractions[surf] = float(v)

        hb = entry.get("height_band_3d") or entry.get("height_band_vg") or entry.get("height_band") or {}
        zc = hb.get("z_center_m") or {}
        zb = hb.get("z_bottom_m") or {}
        e.z_center_p5 = zc.get("p5")
        e.z_center_p95 = zc.get("p95")
        e.z_bottom_p5 = zb.get("p5")
        e.z_bottom_p95 = zb.get("p95")

        e.cooccurrence = entry.get("cooccurrence", {})

        orient_rel_data = entry.get("orientation_relationships", {})
        if orient_rel_data:
            orel = OrientationRelationships()

            btw = orient_rel_data.get("back_to_wall", {})
            if btw:
                orel.back_to_wall_fraction = _safe_float(btw.get("fraction"), 0.0)
                orel.back_to_wall_mean_angle = _safe_float(btw.get("mean_angle"), 90.0)
                orel.back_to_wall_n_samples = int(_safe_float(btw.get("n_samples"), 0))

            fc = orient_rel_data.get("faces_center", {})
            if fc:
                orel.faces_center_fraction = _safe_float(fc.get("fraction"), 0.0)
                orel.faces_center_mean_angle = _safe_float(fc.get("mean_angle"), 90.0)
                orel.faces_center_n_samples = int(_safe_float(fc.get("n_samples"), 0))

            fp = orient_rel_data.get("faces_pair", {})
            if fp:
                cleaned_fp: Dict[str, Dict[str, float]] = {}
                for other_cat, pair_stats in fp.items():
                    if not isinstance(pair_stats, dict):
                        continue
                    frac = _safe_float(pair_stats.get("fraction"), 0.0)
                    mean_angle = _safe_float(pair_stats.get("mean_angle"), 90.0)
                    mean_dist = _safe_float(pair_stats.get("mean_dist"), 2.0)
                    n_samples = int(_safe_float(pair_stats.get("n_samples"), 0))
                    cleaned_fp[other_cat] = {
                        "fraction": frac,
                        "mean_angle": mean_angle,
                        "mean_dist": mean_dist,
                        "n_samples": n_samples,
                    }
                orel.faces_pair = cleaned_fp

            e.orientation_rel = orel

        ontology[cat] = e

    return ontology


def load_verifier_config(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def build_config_from_task(task_config: dict) -> Dict[str, Any]:
    config: Dict[str, Any] = {}

    desc = task_config.get("task_description", "")
    for candidate in ["bedroom", "living_room", "living room", "kitchen",
                      "bathroom", "dining_room", "dining room", "office",
                      "library", "hallway", "laundry_room", "laundry room",
                      "restaurant", "cafe"]:
        if candidate in desc.lower():
            config["domain"] = candidate.replace(" ", "_")
            break
    if "domain" not in config:
        config["domain"] = "bedroom"

    assets = task_config.get("assets", {})
    if assets:
        first_key = next(iter(assets), "")
        base = first_key.rsplit("-", 1)[0] if "-" in first_key else first_key
        is_uid_based = len(base) == 32 and all(c in "0123456789abcdef" for c in base)

        if not is_uid_based:
            config["valid_objects"] = {cat: {"count": cnt} for cat, cnt in assets.items()}
            config["expected_assets"] = dict(assets)

    rd = task_config.get("room_dimension", {})
    if rd:
        config["room_dimension_x"] = rd.get("room_dimension_x", 5)
        config["room_dimension_y"] = rd.get("room_dimension_y", 5)

    boundary = task_config.get("boundary", {})
    vertices = boundary.get("floor_vertices", [])
    if vertices:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        config["room_dimension_x"] = max(xs) - min(xs)
        config["room_dimension_y"] = max(ys) - min(ys)

    return config


class RuleChecker(ABC):
    @property
    @abstractmethod
    def rule_name(self) -> str:
        ...

    @abstractmethod
    def check(
        self,
        objects: List[SceneObject],
        room: RoomInfo,
        ontology: Dict[str, OntologyEntry],
        config: Dict[str, Any],
    ) -> CheckResult:
        ...


class ScaleVerifier(RuleChecker):
    rule_name = "scale"

    HARD_FACTOR = 2.0
    ABS_TOL = 0.05

    def check(self, objects, room, ontology, config):
        violations = []
        successes = []
        n_checked = 0
        n_passed = 0

        for obj in objects:
            entry = ontology.get(obj.category)
            if entry is None:
                continue

            axes = [
                ("width", obj.width, entry.dim_width_min, entry.dim_width_max),
                ("depth", obj.depth, entry.dim_depth_min, entry.dim_depth_max),
                ("height", obj.height, entry.dim_height_min, entry.dim_height_max)
            ]

            checkable = [(a, v, lo, hi) for a, v, lo, hi in axes
                         if lo is not None and hi is not None and v > 0]
            if not checkable:
                continue

            n_checked += 1
            obj_violations = []
            worst_severity = None
            pass_details = []

            for axis_name, actual, lo, hi in checkable:
                lo_soft = lo - self.ABS_TOL
                hi_soft = hi + self.ABS_TOL
                lo_hard = lo / self.HARD_FACTOR
                hi_hard = hi * self.HARD_FACTOR

                if actual < lo_hard or actual > hi_hard:
                    obj_violations.append(
                        f"{axis_name}={actual:.3f}m (absurd, expected {lo:.3f}–{hi:.3f}m)"
                    )
                    worst_severity = Severity.HARD
                elif actual < lo_soft or actual > hi_soft:
                    obj_violations.append(
                        f"{axis_name}={actual:.3f}m (unusual, expected {lo:.3f}–{hi:.3f}m)"
                    )
                    if worst_severity is None:
                        worst_severity = Severity.SOFT
                else:
                    pass_details.append(
                        {"axis": axis_name, "actual": actual, "expected_p5": lo, "expected_p95": hi}
                    )

            if obj_violations:
                violations.append(Violation(
                    rule=self.rule_name,
                    severity=worst_severity,
                    object_id=obj.obj_id,
                    message=(
                        f"{obj.category} '{obj.obj_id}' scale issues: "
                        f"{'; '.join(obj_violations)}"
                    ),
                    details={"axes_flagged": obj_violations},
                ))
            else:
                n_passed += 1
                axis_order = {"width": 0, "depth": 1, "height": 2}
                axis_parts = []
                for d in sorted(pass_details, key=lambda x: axis_order.get(x["axis"], 99)):
                    axis = d["axis"]
                    actual = float(d["actual"])
                    lo = float(d["expected_p5"])
                    hi = float(d["expected_p95"])
                    axis_parts.append(
                        f"{axis}={actual:.3f}m (expected {lo:.3f}–{hi:.3f}m)"
                    )

                msg_suffix = " | " + "; ".join(axis_parts) if axis_parts else ""

                successes.append(Success(
                    rule=self.rule_name,
                    object_id=obj.obj_id,
                    message=f"{obj.category} '{obj.obj_id}' scale OK{msg_suffix}",
                    details={"axes": pass_details},
                ))

        return CheckResult(self.rule_name, n_checked, n_passed, violations, successes)


class OverlapVerifier(RuleChecker):
    rule_name = "overlap"

    TOLERANCE = 0.02

    def check(self, objects, room, ontology, config):
        violations = []
        n_checked = 0
        n_passed = 0

        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                n_checked += 1
                overlap = self._overlap_amount(a, b)
                if overlap > self.TOLERANCE:
                    sev = Severity.HARD if overlap > 0.2 else Severity.SOFT
                    violations.append(Violation(
                        rule=self.rule_name,
                        severity=sev,
                        object_id=f"{a.obj_id},{b.obj_id}",
                        message=(
                            f"{a.category} '{a.obj_id}' overlaps with "
                            f"{b.category} '{b.obj_id}' by {overlap:.3f}m"
                        ),
                        details={"obj_a": a.obj_id, "obj_b": b.obj_id,
                                 "overlap": overlap},
                    ))
                else:
                    n_passed += 1

        return CheckResult(self.rule_name, n_checked, n_passed, violations)

    @staticmethod
    def _overlap_amount(a: SceneObject, b: SceneObject) -> float:
        a_box = a.bbox_2d
        b_box = b.bbox_2d
        ox = min(a_box[2], b_box[2]) - max(a_box[0], b_box[0])
        oy = min(a_box[3], b_box[3]) - max(a_box[1], b_box[1])
        if ox > 0 and oy > 0:
            return min(ox, oy)
        return 0.0

class OBBOverlapVerifier(RuleChecker):
    rule_name = "overlap_obb"

    TOLERANCE = 0.05
    HARD_OVERLAP_THRESHOLD = 0.15

    def check(self, objects, room, ontology, config):
        violations: List[Violation] = []
        successes: List[Success] = []
        n_checked = 0
        n_passed = 0

        for i, a in enumerate(objects):
            for b in objects[i + 1:]:
                n_checked += 1
                penetration = self._obb_penetration(a, b)
                if penetration > self.TOLERANCE:
                    sev = (Severity.HARD
                           if penetration > self.HARD_OVERLAP_THRESHOLD
                           else Severity.SOFT)
                    violations.append(Violation(
                        rule=self.rule_name, severity=sev,
                        object_id=f"{a.obj_id},{b.obj_id}",
                        message=(
                            f"{a.category} '{a.obj_id}' overlaps (OBB) with "
                            f"{b.category} '{b.obj_id}' by {penetration:.3f}m"
                        ),
                        details={
                            "obj_a": a.obj_id, "obj_b": b.obj_id,
                            "penetration_m": round(penetration, 4),
                            "method": "obb_sat",
                        },
                    ))
                else:
                    n_passed += 1
                    successes.append(Success(
                        rule=self.rule_name,
                        object_id=f"{a.obj_id},{b.obj_id}",
                        message=(
                            f"{a.category} '{a.obj_id}' and "
                            f"{b.category} '{b.obj_id}' do not overlap (OBB)"
                        ),
                        details={
                            "obj_a": a.obj_id, "obj_b": b.obj_id,
                            "penetration_m": round(penetration, 4),
                            "method": "obb_sat",
                        },
                    ))

        return CheckResult(self.rule_name, n_checked, n_passed, violations, successes)

    @staticmethod
    def _obb_corners(obj: SceneObject) -> List[Tuple[float, float]]:
        hw = obj.width / 2.0
        hd = obj.depth / 2.0
        angle_rad = math.radians(obj.yaw)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        local_corners = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
        world_corners = []
        for lx, ly in local_corners:
            wx = obj.x + lx * cos_a - ly * sin_a
            wy = obj.y + lx * sin_a + ly * cos_a
            world_corners.append((wx, wy))
        return world_corners

    @staticmethod
    def _get_axes(corners: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        axes = []
        for i in range(2):
            x1, y1 = corners[i]
            x2, y2 = corners[i + 1]
            edge_x = x2 - x1
            edge_y = y2 - y1
            length = math.hypot(edge_x, edge_y)
            if length < 1e-12:
                continue
            axes.append((-edge_y / length, edge_x / length))
        return axes

    @staticmethod
    def _project(corners, axis):
        dots = [c[0] * axis[0] + c[1] * axis[1] for c in corners]
        return min(dots), max(dots)

    @classmethod
    def _obb_penetration(cls, a: SceneObject, b: SceneObject) -> float:
        corners_a = cls._obb_corners(a)
        corners_b = cls._obb_corners(b)
        axes = cls._get_axes(corners_a) + cls._get_axes(corners_b)
        min_overlap = float("inf")
        for axis in axes:
            min_a, max_a = cls._project(corners_a, axis)
            min_b, max_b = cls._project(corners_b, axis)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap <= 0:
                return 0.0
            if overlap < min_overlap:
                min_overlap = overlap
        return min_overlap

class OrientationVerifier(RuleChecker):
    rule_name = "orientation"

    BACK_TO_WALL_THRESHOLD = 0.30
    FACES_CENTER_THRESHOLD = 0.30
    FACES_PAIR_THRESHOLD = 0.30
    FACES_PAIR_MAX_DIST = 3.0

    SOFT_ANGLE = 75.0
    HARD_ANGLE = 150.0

    MIN_SAMPLES = 10

    EXCLUDED_CATEGORIES = {
        "stool",
        "table",
        "floor_lamp",
        "coffee_table",
        "side_table",
        "mirror",
    }

    FALLBACK_BACK_TO_WALL = {
        "bookshelf", "wardrobe", "dresser", "cabinet",
        "tv_stand", "shelf",
    }

    FALLBACK_FACES_PAIR = {
        ("sofa", "tv_stand"): 3.0,
        ("armchair", "tv_stand"): 3.0,
        ("sofa", "coffee_table"): 2.0,
        ("armchair", "coffee_table"): 2.0,
        ("chair", "desk"): 2.0,
        ("office_chair", "desk"): 2.0,
        ("bed", "side_table"): 1.5
    }

    def check(self, objects, room, ontology, config):
        violations = []
        successes = []
        n_checked = 0
        n_passed = 0

        def _is_nan_yaw(yaw_val) -> bool:
            if yaw_val is None:
                return True
            try:
                return math.isnan(float(yaw_val))
            except (TypeError, ValueError):
                return True

        if any(_is_nan_yaw(obj.yaw) for obj in objects):
            return CheckResult(self.rule_name, 0, 0, violations, successes)

        walls = self._wall_segments(room)

        for obj in objects:
            cat = obj.category

            if cat in self.EXCLUDED_CATEGORIES:
                continue

            entry = ontology.get(cat)
            orel = entry.orientation_rel if entry else OrientationRelationships()

            obj_successes = []
            obj_violations = []
            any_check_run = False

            use_back_to_wall = False
            if orel.back_to_wall_n_samples >= self.MIN_SAMPLES:
                use_back_to_wall = orel.back_to_wall_fraction >= self.BACK_TO_WALL_THRESHOLD
            else:
                use_back_to_wall = cat in self.FALLBACK_BACK_TO_WALL

            if use_back_to_wall and walls:
                nearest_wall_normal = self._nearest_wall_normal(obj, walls)
                if nearest_wall_normal is not None:
                    any_check_run = True
                    result = self._check_back_toward(obj, nearest_wall_normal)
                    frac = orel.back_to_wall_fraction if orel.back_to_wall_n_samples >= self.MIN_SAMPLES else 0.0
                    if result == "pass":
                        obj_successes.append(Success(
                            rule=self.rule_name,
                            object_id=obj.obj_id,
                            message=f"{cat} '{obj.obj_id}' orientation OK: back_to_wall (yaw={obj.yaw:.0f}°)",
                            details={
                                "check_type": "back_to_wall",
                                "yaw": obj.yaw,
                                "data_fraction": orel.back_to_wall_fraction,
                                "n_samples": orel.back_to_wall_n_samples,
                            },
                        ))
                    else:
                        sev = Severity.HARD if result == "hard" else Severity.SOFT
                        frac_info = ""
                        if orel.back_to_wall_n_samples >= self.MIN_SAMPLES:
                            frac_info = f" (back_to_wall={orel.back_to_wall_fraction:.0%} in data)"
                        obj_violations.append((frac, Violation(
                            rule=self.rule_name, severity=sev,
                            object_id=obj.obj_id,
                            message=f"{cat} '{obj.obj_id}' should have back against wall (yaw={obj.yaw:.0f}°){frac_info}",
                            details={"check_type": "back_to_wall", "yaw": obj.yaw,
                                     "data_fraction": orel.back_to_wall_fraction},
                        )))

            use_faces_center = False
            if orel.faces_center_n_samples >= self.MIN_SAMPLES:
                use_faces_center = orel.faces_center_fraction >= self.FACES_CENTER_THRESHOLD

            if use_faces_center and room.bounds:
                any_check_run = True
                cx = (room.bounds[0] + room.bounds[2]) / 2.0
                cy = (room.bounds[1] + room.bounds[3]) / 2.0
                result = self._check_faces_point(obj, cx, cy)
                if result == "pass":
                    obj_successes.append(Success(
                        rule=self.rule_name,
                        object_id=obj.obj_id,
                        message=f"{cat} '{obj.obj_id}' orientation OK: faces_center (yaw={obj.yaw:.0f}°)",
                        details={
                            "check_type": "faces_center",
                            "yaw": obj.yaw,
                            "data_fraction": orel.faces_center_fraction,
                            "n_samples": orel.faces_center_n_samples,
                        },
                    ))
                else:
                    sev = Severity.HARD if result == "hard" else Severity.SOFT
                    obj_violations.append((orel.faces_center_fraction, Violation(
                        rule=self.rule_name, severity=sev,
                        object_id=obj.obj_id,
                        message=(f"{cat} '{obj.obj_id}' should face room center "
                                 f"(yaw={obj.yaw:.0f}°, faces_center={orel.faces_center_fraction:.0%})"),
                        details={"check_type": "faces_center", "yaw": obj.yaw,
                                 "data_fraction": orel.faces_center_fraction},
                    )))

            has_cat_faces_pair = bool(orel.faces_pair)

            if has_cat_faces_pair:
                for other_cat, pair_stats in orel.faces_pair.items():
                    if not isinstance(pair_stats, dict):
                        continue
                    frac = _safe_float(pair_stats.get("fraction"), 0.0)
                    n_samp = int(_safe_float(pair_stats.get("n_samples"), 0))
                    if frac < self.FACES_PAIR_THRESHOLD or n_samp < self.MIN_SAMPLES:
                        continue

                    max_dist = 6.0
                    nearby = self._find_nearest_by_category(
                        obj, objects, {other_cat},
                        excluded_objects=self.EXCLUDED_CATEGORIES,
                        max_dist=max_dist,
                    )
                    if nearby is None:
                        continue
                    
                    any_check_run = True
                    result = self._check_faces_point(obj, nearby.x, nearby.y)

                    if result == "pass":
                        obj_successes.append(Success(
                            rule=self.rule_name,
                            object_id=obj.obj_id,
                            message=(f"{cat} '{obj.obj_id}' orientation OK: faces_pair "
                                     f"(target={other_cat}, yaw={obj.yaw:.0f}°)"),
                            details={
                                "check_type": "faces_pair",
                                "target_cat": other_cat,
                                "target_obj_id": nearby.obj_id,
                                "yaw": obj.yaw,
                                "data_fraction": frac,
                                "n_samples": n_samp,
                                "max_dist": max_dist,
                            },
                        ))
                    else:
                        sev = Severity.HARD if result == "hard" else Severity.SOFT
                        obj_violations.append((frac, Violation(
                            rule=self.rule_name, severity=sev,
                            object_id=obj.obj_id,
                            message=(f"{cat} '{obj.obj_id}' near {nearby.category} '{nearby.obj_id}' "
                                     f"but doesn't face it (yaw={obj.yaw:.0f}°, faces_{other_cat}={frac:.0%})"),
                            details={"check_type": "faces_pair", "target_cat": other_cat,
                                     "yaw": obj.yaw, "data_fraction": frac},
                        )))

            else:
                for (seat_cat, table_cat), max_dist in self.FALLBACK_FACES_PAIR.items():
                    if cat != seat_cat:
                        continue
                    nearby = self._find_nearest_by_category(obj, objects, {table_cat}, max_dist=max_dist)
                    if nearby is None:
                        continue

                    any_check_run = True
                    result = self._check_faces_point(obj, nearby.x, nearby.y)
                    if result == "pass":
                        obj_successes.append(Success(
                            rule=self.rule_name,
                            object_id=obj.obj_id,
                            message=(f"{cat} '{obj.obj_id}' orientation OK: faces_pair_fallback "
                                     f"(target={table_cat}, yaw={obj.yaw:.0f}°)"),
                            details={
                                "check_type": "faces_pair_fallback",
                                "target_cat": table_cat,
                                "target_obj_id": nearby.obj_id,
                                "yaw": obj.yaw,
                                "max_dist": max_dist,
                            },
                        ))
                    else:
                        sev = Severity.HARD if result == "hard" else Severity.SOFT
                        obj_violations.append((0.0, Violation(
                            rule=self.rule_name, severity=sev,
                            object_id=obj.obj_id,
                            message=(f"{cat} '{obj.obj_id}' near {nearby.category} "
                                     f"but doesn't face it (yaw={obj.yaw:.0f}°)"),
                            details={"check_type": "faces_pair_fallback", "yaw": obj.yaw},
                        )))
                    break

            if not any_check_run:
                continue

            n_checked += 1

            if obj_successes:
                n_passed += 1
                successes.extend(obj_successes)
            else:
                obj_violations.sort(key=lambda v: v[0], reverse=True)
                worst_frac, worst_violation = obj_violations[0]
                violations.append(worst_violation)
        return CheckResult(self.rule_name, n_checked, n_passed, violations, successes)

    def _compute_target_angle(self, obj: SceneObject, dy: float, dx: float) -> float:
        if obj._source == "holodeck":
            return math.degrees(math.atan2(dy, dx))
        else:
            ang = math.degrees(math.atan2(dy, dx))
            return (ang + 90.0) % 360.0

    def _check_faces_point(self, obj, tx, ty):
        dx = tx - obj.x
        dy = ty - obj.y
        if abs(dx) < 0.01 and abs(dy) < 0.01:
            return "pass"

        target_angle = self._compute_target_angle(obj, dy, dx)
        return self._angle_check(obj.yaw, target_angle)

    def _check_back_toward(self, obj, wall_normal):
        target_angle = self._compute_target_angle(obj, wall_normal[1], wall_normal[0])
        return self._angle_check(obj.yaw, target_angle)

    def _angle_check(self, yaw_deg, target_deg):
        diff = self._angle_diff(yaw_deg, target_deg)
        if diff <= self.SOFT_ANGLE:
            return "pass"
        elif diff >= self.HARD_ANGLE:
            return "hard"
        return "soft"

    @staticmethod
    def _angle_diff(a, b):
        d = (a - b) % 360
        if d > 180:
            d = 360 - d
        return d
    
    def _atan2_to_yaw_convention(self, dy, dx) -> float:
        ang = math.degrees(math.atan2(dy, dx))
        return (ang + 90.0) % 360.0

    @staticmethod
    def _wall_segments(room):
        if not room.bounds:
            return []
        x0, y0, x1, y1 = room.bounds
        return [
            ((x0, (y0 + y1) / 2), (1, 0)),
            ((x1, (y0 + y1) / 2), (-1, 0)),
            (((x0 + x1) / 2, y0), (0, -1)),
            (((x0 + x1) / 2, y1), (0, +1)),
        ]

    def _nearest_wall_normal(self, obj, walls, room=None):
        best_dist = float("inf")
        best_normal = None
        for midpoint, normal in walls:
            dx = obj.x - midpoint[0]
            dy = obj.y - midpoint[1]
            perp_dist = abs(dx * normal[0] + dy * normal[1])
            if perp_dist < best_dist:
                best_dist = perp_dist
                best_normal = normal
        return best_normal

    @staticmethod
    def _find_nearest_by_category(obj, all_objects, target_cats, excluded_objects=None, max_dist=6.0):
        if excluded_objects is None:
            excluded_objects = set()
        best_obj = None
        best_d = max_dist
        for other in all_objects:
            if (other.obj_id == obj.obj_id
                    or other.category not in target_cats
                    or other.category in excluded_objects):
                continue
            d = math.hypot(obj.x - other.x, obj.y - other.y)
            if d < best_d:
                best_d = d
                best_obj = other
        return best_obj


class CooccurrenceVerifier(RuleChecker):
    rule_name = "cooccurrence"

    RARE_THRESHOLD = 0.01
    ABSENT_THRESHOLD = 0.001

    STRONG_COOCCUR_THRESHOLD = 0.20
    FUNCTIONAL_COOCCUR_THRESHOLD = 0.70

    FUNCTIONAL_MAX_DIST = 2.0
    STRONG_MAX_DIST = 3.5

    FUNCTIONAL_PAIRS = {
        ("desk", "chair"): 1.5,
        ("desk", "office_chair"): 1.5,
        ("dining_table", "chair"): 1.5,
        ("dining_table", "dining_chair"): 1.5,
        ("bed", "nightstand"): 1.5,
        ("tv", "tv_stand"): 1.0,
        ("sofa", "coffee_table"): 2.0,
    }

    SKIP_CATS = {"door", "window", "wall", "floor", "ceiling",
                 "pendant_lamp", "curtain"}

    def check(self, objects, room, ontology, config):
        violations = []
        successes = []
        n_checked = 0
        n_passed = 0

        cat_to_objs: Dict[str, List[SceneObject]] = {}
        for obj in objects:
            if obj.category not in self.SKIP_CATS:
                cat_to_objs.setdefault(obj.category, []).append(obj)

        present_cats = set(cat_to_objs.keys())
        cat_list = sorted(present_cats)

        proximity_checked: Set[Tuple[str, str]] = set()

        for i, cat_a in enumerate(cat_list):
            entry_a = ontology.get(cat_a)
            for cat_b in cat_list[i + 1:]:
                entry_b = ontology.get(cat_b)

                if entry_a is None or entry_b is None:
                    continue

                frac_a_b = self._cooccur_frac(entry_a, cat_b)
                frac_b_a = self._cooccur_frac(entry_b, cat_a)
                best_frac = max(frac_a_b, frac_b_a)

                objs_a = cat_to_objs[cat_a]
                objs_b = cat_to_objs[cat_b]

                if best_frac >= self.RARE_THRESHOLD:
                    for obj_a in objs_a:
                        for obj_b in objs_b:
                            n_checked += 1
                            n_passed += 1
                            successes.append(Success(
                                rule=self.rule_name,
                                object_id=f"{obj_a.obj_id},{obj_b.obj_id}",
                                message=f"'{cat_a}' and '{cat_b}' plausibility OK (frac={best_frac:.2f})",
                                details={
                                    "cat_a": cat_a,
                                    "cat_b": cat_b,
                                    "fraction": round(best_frac, 3),
                                    "frac_a_b": round(frac_a_b, 3),
                                    "frac_b_a": round(frac_b_a, 3),
                                    "check_type": "plausibility_pass",
                                },
                            ))

                    if best_frac >= self.STRONG_COOCCUR_THRESHOLD:
                        self._check_proximity_data_driven(
                            cat_a, cat_b, best_frac,
                            cat_to_objs, violations, successes,
                            n_checked_ref=[n_checked],
                            n_passed_ref=[n_passed],
                        )
                        proximity_checked.add(tuple(sorted((cat_a, cat_b))))

                else:
                    for obj_a in objs_a:
                        for obj_b in objs_b:
                            n_checked += 1
                            violations.append(Violation(
                                rule=self.rule_name,
                                severity=Severity.INFO,
                                object_id=f"{obj_a.obj_id},{obj_b.obj_id}",
                                message=(
                                    f"'{cat_a}' and '{cat_b}' are rarely seen "
                                    f"together (frac: {frac_a_b:.3f} / {frac_b_a:.3f})"
                                ),
                                details={
                                    "cat_a": cat_a, "cat_b": cat_b,
                                    "obj_a_id": obj_a.obj_id,
                                    "obj_b_id": obj_b.obj_id,
                                    "frac_a_b": round(frac_a_b, 4),
                                    "frac_b_a": round(frac_b_a, 4),
                                    "check_type": "plausibility",
                                },
                            ))

        for (cat_a, cat_b), max_dist in self.FUNCTIONAL_PAIRS.items():
            pair_key = tuple(sorted([cat_a, cat_b]))
            if pair_key in proximity_checked:
                continue
            if cat_a not in present_cats or cat_b not in present_cats:
                continue

            n_checked += 1
            min_d = self._min_distance_between(cat_to_objs[cat_a],
                                                cat_to_objs[cat_b])
            obj_a0 = cat_to_objs[cat_a][0]
            obj_b0 = cat_to_objs[cat_b][0]
            if min_d <= max_dist:
                n_passed += 1
                successes.append(Success(
                    rule=self.rule_name,
                    object_id=f"{obj_a0.obj_id},{obj_b0.obj_id}",
                    message=(
                        f"'{cat_a}' and '{cat_b}' functional proximity OK "
                        f"(dist={min_d:.2f}m ≤ {max_dist:.1f}m)"
                    ),
                    details={
                        "cat_a": cat_a,
                        "cat_b": cat_b,
                        "distance": round(min_d, 3),
                        "max_distance": max_dist,
                        "check_type": "functional_proximity_pass",
                    },
                ))
            else:
                violations.append(Violation(
                    rule=self.rule_name,
                    severity=Severity.SOFT,
                    object_id=f"{obj_a0.obj_id},{obj_b0.obj_id}",
                    message=(
                        f"'{cat_a}' and '{cat_b}' are a functional pair "
                        f"but placed {min_d:.2f}m apart "
                        f"(should be within {max_dist:.1f}m)"
                    ),
                    details={
                        "cat_a": cat_a, "cat_b": cat_b,
                        "distance": round(min_d, 3),
                        "max_distance": max_dist,
                        "check_type": "functional_proximity",
                    },
                ))

        return CheckResult(self.rule_name, n_checked, n_passed, violations, successes)

    @staticmethod
    def _cooccur_frac(entry: Optional[OntologyEntry], other_cat: str) -> float:
        if entry is None or not entry.cooccurrence:
            return 0.0

        co_val = entry.cooccurrence.get(other_cat)
        if co_val is None:
            return 0.0

        if isinstance(co_val, dict):
            return _safe_float(co_val.get("p_b_given_a"), 0.0)

        if isinstance(co_val, (int, float)) and entry.total_scene_count > 0:
            return float(co_val) / entry.total_scene_count

        return 0.0

    def _check_proximity_data_driven(
        self, cat_a, cat_b, frac, cat_to_objs, violations, successes,
        n_checked_ref, n_passed_ref,
    ):
        if frac >= self.FUNCTIONAL_COOCCUR_THRESHOLD:
            max_dist = self.FUNCTIONAL_MAX_DIST
            severity = Severity.SOFT
        else:
            max_dist = self.STRONG_MAX_DIST
            severity = Severity.INFO

        objs_a = cat_to_objs.get(cat_a, [])
        objs_b = cat_to_objs.get(cat_b, [])
        if not objs_a or not objs_b:
            return

        min_d = self._min_distance_between(objs_a, objs_b)
        if min_d > max_dist:
            violations.append(Violation(
                rule=self.rule_name,
                severity=severity,
                object_id=f"{objs_a[0].obj_id},{objs_b[0].obj_id}",
                message=(
                    f"'{cat_a}' and '{cat_b}' co-occur in "
                    f"{frac:.0%} of scenes but are placed "
                    f"{min_d:.2f}m apart (expected within {max_dist:.1f}m)"
                ),
                details={
                    "cat_a": cat_a, "cat_b": cat_b,
                    "distance": round(min_d, 3),
                    "max_distance": max_dist,
                    "fraction": round(frac, 3),
                    "check_type": "data_driven_proximity",
                },
            ))
        else:
            n_passed_ref[0] += 1
            successes.append(Success(
                rule=self.rule_name,
                object_id=f"{objs_a[0].obj_id},{objs_b[0].obj_id}",
                message=(
                    f"'{cat_a}' and '{cat_b}' proximity OK "
                    f"(dist={min_d:.2f}m ≤ {max_dist:.1f}m)"
                ),
                details={
                    "cat_a": cat_a,
                    "cat_b": cat_b,
                    "distance": round(min_d, 3),
                    "max_distance": max_dist,
                    "fraction": round(frac, 3),
                    "check_type": "data_driven_proximity_pass",
                },
            ))

    @staticmethod
    def _min_distance_between(objs_a, objs_b) -> float:
        min_d = float("inf")
        for a in objs_a:
            for b in objs_b:
                d = math.hypot(a.x - b.x, a.y - b.y)
                if d < min_d:
                    min_d = d
        return min_d


class CompletenessVerifier(RuleChecker):
    rule_name = "completeness"

    def check(self, objects, room, ontology, config):
        violations = []
        successes = []
        n_checked = 0
        n_passed = 0

        expected = config.get("expected_assets", {})
        if not expected:
            return CheckResult(self.rule_name, 0, 0, violations, successes)

        cat_counts: Dict[str, int] = {}
        for obj in objects:
            cat = obj.category.lower()
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        for raw_cat, expected_count in expected.items():
            raw_cat_norm = raw_cat.lower().replace(" ", "_")
            canonical = normalize_category(raw_cat_norm)

            matching_names = {
                name.replace(" ", "_")
                for name, canon in CATEGORY_ALIASES.items()
                if canon == canonical
            }
            matching_names.add(canonical)

            actual_count = sum(
                count for name, count in cat_counts.items()
                if name in matching_names
            )

            matched = min(actual_count, expected_count)
            missing = max(expected_count - actual_count, 0)
            extra = max(actual_count - expected_count, 0)

            for i in range(matched):
                n_checked += 1
                n_passed += 1
                successes.append(Success(
                    rule=self.rule_name,
                    object_id="scene",
                    message=(
                        f"'{raw_cat}' (canonical '{canonical}') object "
                        f"{i+1}/{expected_count} found."
                    ),
                    details={
                        "raw": raw_cat,
                        "canonical": canonical,
                        "expected": expected_count,
                        "actual": actual_count,
                        "instance": i + 1,
                        "aliases_checked": list(matching_names),
                    },
                ))

            for i in range(missing):
                n_checked += 1
                sev = Severity.HARD if actual_count == 0 else Severity.SOFT
                violations.append(Violation(
                    rule=self.rule_name, severity=sev,
                    object_id="scene",
                    message=(
                        f"'{raw_cat}' (canonical '{canonical}') object "
                        f"{matched + i + 1}/{expected_count} missing."
                    ),
                    details={
                        "raw": raw_cat,
                        "canonical": canonical,
                        "expected": expected_count,
                        "actual": actual_count,
                        "missing_instance": matched + i + 1,
                        "aliases_checked": list(matching_names),
                    },
                ))

            for i in range(extra):
                n_checked += 1
                violations.append(Violation(
                    rule=self.rule_name, severity=Severity.SOFT,
                    object_id="scene",
                    message=(
                        f"'{raw_cat}' (canonical '{canonical}') has extra "
                        f"object {expected_count + i + 1} "
                        f"(expected {expected_count}, found {actual_count})."
                    ),
                    details={
                        "raw": raw_cat,
                        "canonical": canonical,
                        "expected": expected_count,
                        "actual": actual_count,
                        "extra_instance": expected_count + i + 1,
                    },
                ))

        expected_canonicals = set()
        for raw_cat in expected.keys():
            raw_cat_norm = raw_cat.lower().replace(" ", "_")
            expected_canonicals.add(normalize_category(raw_cat_norm))

        for cat, count in cat_counts.items():
            canonical = normalize_category(cat)
            if canonical not in expected_canonicals:
                for i in range(count):
                    n_checked += 1
                    violations.append(Violation(
                        rule=self.rule_name, severity=Severity.SOFT,
                        object_id="scene",
                        message=(
                            f"Unexpected object '{cat}' (canonical '{canonical}') "
                            f"instance {i+1}/{count} not in expected assets."
                        ),
                        details={
                            "canonical": canonical,
                            "actual": count,
                            "expected": 0,
                            "instance": i + 1,
                        },
                    ))

        return CheckResult(self.rule_name, n_checked, n_passed, violations, successes)


class SemanticVerifier:
    def __init__(
        self,
        ontology: Dict[str, OntologyEntry],
        config: Dict[str, Any],
        checkers: Optional[List[RuleChecker]] = None,
        category_mesh_dims: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ):
        self.ontology = ontology
        self.config = config
        self.category_mesh_dims = category_mesh_dims
        self.checkers = checkers or [
            ScaleVerifier(),
            OverlapVerifier(),
            OBBOverlapVerifier(),
            OrientationVerifier(),
            CooccurrenceVerifier(),
            CompletenessVerifier(),
        ]

    @classmethod
    def from_files(
        cls,
        ontology_path: str,
        config_path: Optional[str] = None,
        task_config_path: Optional[str] = None,
        task_config: Optional[dict] = None,
        asset_dir: Optional[str] = None,
        checkers: Optional[List[RuleChecker]] = None,
    ) -> "SemanticVerifier":
        ontology = load_ontology(ontology_path)

        config = {}
        if config_path:
            config = load_verifier_config(config_path)

        if task_config_path:
            with open(task_config_path, "r") as f:
                task_config = json.load(f)

        if task_config:
            if is_layoutvlm_task_config(task_config):
                task_derived = build_config_from_layoutvlm_task(task_config)
            else:
                task_derived = build_config_from_task(task_config)
            for k, v in task_derived.items():
                if k not in config:
                    config[k] = v

        category_mesh_dims = None
        if asset_dir:
            if asset_dir.endswith(".json"):
                category_mesh_dims = build_category_mesh_dims_from_json(asset_dir)

        return cls(ontology, config, checkers, category_mesh_dims)

    def verify(
        self,
        scene_json: Dict[str, Any],
        task_config: Optional[dict] = None,
    ) -> VerificationResult:
        if is_layoutvlm_scene(scene_json):
            layoutvlm_expected = build_expected_assets_from_layoutvlm_scene(scene_json)

            scene_json = convert_layoutvlm_scene(
                scene_json,
                task_config=task_config,
                category_mesh_dims=self.category_mesh_dims,
                ontology=self.ontology,
            )
        elif is_2d_scene(scene_json):
            layoutvlm_expected = None
            scene_json = convert_2d_scene(
                scene_json, ontology=self.ontology, task_config=task_config,
                category_mesh_dims=self.category_mesh_dims,
            )
        else:
            layoutvlm_expected = None

        config = dict(self.config)
        if task_config:
            if is_layoutvlm_task_config(task_config):
                derived = build_config_from_layoutvlm_task(
                    task_config, category_mesh_dims=self.category_mesh_dims
                )
            else:
                derived = build_config_from_task(task_config)
            for k, v in derived.items():
                if k not in config:
                    config[k] = v

        if layoutvlm_expected and "expected_assets" not in config:
            config["expected_assets"] = layoutvlm_expected

        objects, room = parse_scene(scene_json)

        rule_results: Dict[str, CheckResult] = {}
        all_violations: List[Violation] = []
        all_successes: List[Success] = []

        for checker in self.checkers:
            result = checker.check(objects, room, self.ontology, config)
            rule_results[result.rule_name] = result
            all_violations.extend(result.violations)
            all_successes.extend(result.successes)

        return VerificationResult(
            rule_results=rule_results,
            violations=all_violations,
            successes=all_successes,
        )

    def verify_batch(self, scenes, task_config=None):
        return [self.verify(s, task_config) for s in scenes]

    def get_reward_signal(self, scene_json, task_config=None):
        result = self.verify(scene_json, task_config)
        signal = {
            "sp": result.sp_score,
            "sp_weighted": result.sp_score_weighted,
            "n_hard_violations": result.n_hard_violations,
        }
        for name, score in result.by_rule.items():
            signal[f"sp_{name}"] = score
        return signal


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Semantic Plausibility Verifier (2D/3D/LayoutVLM)"
    )
    parser.add_argument("scene", help="Path to scene JSON file")
    parser.add_argument(
        "--ontology", default="ontology_output/object_ontology.json",
        help="Path to object ontology JSON",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to verifier config JSON",
    )
    parser.add_argument(
        "--task-config", default=None,
        help="Path to task/generation config JSON",
    )
    parser.add_argument(
        "--asset-dir", default=None,
        help="Path to mesh dimensions JSON (pre-computed from .glb files)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output as JSON",
    )

    args = parser.parse_args()

    with open(args.scene, "r") as f:
        scene = json.load(f)

    task_config = None
    if args.task_config:
        with open(args.task_config, "r") as f:
            task_config = json.load(f)

    sv = SemanticVerifier.from_files(
        ontology_path=args.ontology,
        config_path=args.config,
        task_config=task_config,
        asset_dir=args.asset_dir,
    )
    result = sv.verify(scene, task_config=task_config)

    if args.json:
        output = {
            "sp_score": result.sp_score,
            "sp_score_weighted": result.sp_score_weighted,
            "by_rule": result.by_rule,
            "n_violations": len(result.violations),
            "n_hard_violations": result.n_hard_violations,
            "violations": [
                {
                    "rule": v.rule,
                    "severity": v.severity.value,
                    "object_id": v.object_id,
                    "message": v.message,
                    "details": v.details,
                }
                for v in result.violations
            ],
            "successes": [
                {
                    "rule": s.rule,
                    "object_id": s.object_id,
                    "message": s.message,
                    "details": s.details,
                }
                for s in result.successes
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(result.summary())


if __name__ == "__main__":
    main()