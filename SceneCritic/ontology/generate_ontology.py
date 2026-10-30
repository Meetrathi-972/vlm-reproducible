from __future__ import annotations

import os
import sys
import json
import csv
import glob
import math
import argparse
import logging
import re
import struct
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BACK_TO_WALL_CATEGORIES: Set[str] = {
    "bed", "sofa", "armchair", "recliner", "dresser", "desk", "tv_stand",
    "cabinet", "wardrobe", "bookshelf", "sideboard", "console_table",
    "dressing_table", "filing_cabinet",
}
FACES_CENTER_CATEGORIES: Set[str] = {
    "sofa", "armchair", "chair", "recliner", "tv", "monitor", "desk",
    "office_chair", "bed",
}
FACES_PAIR_CATEGORIES: Set[str] = {
    "sofa", "armchair", "chair", "office_chair", "recliner", "bed",
    "desk", "tv", "monitor", "dining_table", "coffee_table",
}

CATEGORY_FORWARD_OFFSET: Dict[str, float] = {
}

CATEGORY_ALIASES: Dict[str, str] = { ###ADD MORE ALIASING FOR MISSING CATEGORY IN DATASET
    "chair": "chair", "dining chair": "chair", "desk chair": "chair",
    "office chair": "office_chair", "swivel chair": "office_chair",
    "armchair": "armchair", "lounge chair": "armchair", "accent chair": "armchair",
    "sofa": "sofa", "couch": "sofa", "loveseat": "sofa", "settee": "sofa",
    "sectional": "sofa", "l-shaped sofa": "sofa",
    "stool": "stool", "bar stool": "stool", "counter stool": "stool",
    "bench": "bench", "ottoman": "ottoman", "pouf": "ottoman",
    "recliner": "recliner", "rocking chair": "rocking_chair",
    "bean bag": "bean_bag", "beanbag": "bean_bag",
    "table": "table", "desk": "desk", "writing desk": "desk",
    "computer desk": "desk", "study desk": "desk",
    "dining table": "dining_table", "kitchen table": "dining_table",
    "coffee table": "coffee_table", "cocktail table": "coffee_table",
    "side table": "side_table", "end table": "side_table", "accent table": "side_table",
    "nightstand": "nightstand", "bedside table": "nightstand", "night stand": "nightstand",
    "console table": "console_table", "console": "console_table",
    "dressing table": "dressing_table", "vanity": "dressing_table",
    "bed": "bed", "single bed": "bed", "double bed": "bed",
    "king bed": "bed", "queen bed": "bed", "bunk bed": "bunk_bed",
    "crib": "crib", "baby bed": "crib", "children bed": "bed",
    "wardrobe": "wardrobe", "closet": "wardrobe", "armoire": "wardrobe",
    "bookshelf": "bookshelf", "bookcase": "bookshelf", "shelving": "bookshelf",
    "shelf": "shelf", "wall shelf": "wall_shelf", "floating shelf": "wall_shelf",
    "cabinet": "cabinet", "cupboard": "cabinet", "storage cabinet": "cabinet",
    "dresser": "dresser", "chest of drawers": "dresser", "drawer": "dresser",
    "tv stand": "tv_stand", "media console": "tv_stand", "entertainment center": "tv_stand",
    "sideboard": "sideboard", "buffet": "sideboard", "credenza": "sideboard",
    "filing cabinet": "filing_cabinet", "file cabinet": "filing_cabinet",
    "cabinet/shelf/desk": "cabinet",
    "pier/stool": "stool",
    "king-size bed": "bed",
    "kids bed": "bed",
    "bed frame": "bed",
    "round end table": "side_table",
    "lounge chair/cafe chair/office chair": "chair",
    "dressing chair": "chair",
    "classic chinese chair": "chair",
    "barstool": "stool",
    "three-seat/multi-seat sofa": "sofa",
    "two-seat sofa": "sofa",
    "chaise longue sofa": "sofa",
    "lazy sofa": "sofa",
    "corner/side table": "side_table",
    "sideboard/side cabinet/console table": "sideboard",
    "children cabinet": "cabinet",
    "drawer chest/corner cabinet": "dresser",
    "wine cabinet": "cabinet",
    "bookcase/jewelry armoire": "bookshelf",
    "ceiling lamp": "ceiling_lamp",
    "pendant lamp": "pendant_lamp",
    "floor lamp": "floor_lamp",
    "wall lamp": "wall_lamp",
    "table lamp": "table_lamp",
    "bulb": "lamp",
    "lamp": "lamp", "desk lamp": "table_lamp",
    "standing lamp": "floor_lamp",
    "chandelier": "chandelier",
    "pendant light": "pendant_lamp",
    "sconce": "wall_lamp",
    "tv": "tv", "television": "tv", "monitor": "monitor", "computer monitor": "monitor",
    "laptop": "laptop", "computer": "computer", "desktop computer": "computer",
    "speaker": "speaker", "soundbar": "speaker",
    "printer": "printer",
    "refrigerator": "refrigerator", "fridge": "refrigerator",
    "oven": "oven", "stove": "stove", "cooktop": "stove",
    "microwave": "microwave", "dishwasher": "dishwasher",
    "washing machine": "washing_machine", "washer": "washing_machine",
    "dryer": "dryer", "air conditioner": "air_conditioner",
    "toilet": "toilet", "sink": "sink", "bathroom sink": "sink",
    "bathtub": "bathtub", "tub": "bathtub", "shower": "shower",
    "mirror": "mirror", "bathroom mirror": "mirror",
    "rug": "rug", "carpet": "rug", "area rug": "rug",
    "curtain": "curtain", "drape": "curtain", "window curtain": "curtain",
    "painting": "painting", "picture": "painting", "wall art": "painting",
    "photo frame": "photo_frame", "picture frame": "photo_frame",
    "vase": "vase", "flower vase": "vase",
    "plant": "plant", "potted plant": "plant", "houseplant": "plant",
    "clock": "clock", "wall clock": "clock",
    "pillow": "pillow", "cushion": "pillow", "throw pillow": "pillow",
    "blanket": "blanket", "throw": "blanket",
    "umbrella": "patio_umbrella", "patio umbrella": "patio_umbrella",
    "grill": "grill", "bbq": "grill", "barbecue": "grill",
    "planter": "planter", "flower pot": "planter",
    "outdoor sofa": "outdoor_sofa", "garden bench": "garden_bench",
    "fire pit": "fire_pit", "firepit": "fire_pit",
    "lounge chair": "lounge_chair", "sun lounger": "lounge_chair",
    "chaise lounge": "lounge_chair",
    "door": "door", "window": "window",
    "ceiling fan": "ceiling_fan", "fan": "fan",
    "fireplace": "fireplace",
    "piano": "piano", "keyboard": "keyboard",
    "counter": "counter",
    "shop counter": "counter",
    "checkout counter": "counter",
    "service counter": "counter",
    "display_shelf": "shelf",
    "rack": "shelf",
}

def _tokenize(s: str) -> List[str]:
    return [t for t in re.split(r"[^a-z0-9_]+", s.lower().strip()) if t]

def _alias_tokens(alias: str) -> List[str]:
    return _tokenize(alias.replace("/", " ").replace("-", " "))

def _is_wordish(alias: str) -> bool:
    return bool(re.fullmatch(r"[a-z0-9_\s-]+", alias)) and ("/" not in alias)

def normalize_category(raw_label: str) -> Optional[str]:
    if not raw_label:
        return None
    label = raw_label.strip().lower()
    if not label:
        return None
    if label in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[label]

    label_tokens = _tokenize(label)
    label_token_str = " ".join(label_tokens)

    best_match = None
    best_score = -1

    for alias, canonical in CATEGORY_ALIASES.items():
        if not alias:
            continue

        if not _is_wordish(alias):
            if alias in label:
                score = len(alias)
                if score > best_score:
                    best_score = score
                    best_match = canonical
            continue

        a_tokens = _alias_tokens(alias)
        if not a_tokens:
            continue

        if len(a_tokens) == 1:
            if a_tokens[0] in label_tokens:
                score = 10 + len(a_tokens[0])
                if score > best_score:
                    best_score = score
                    best_match = canonical
            continue

        a_str = " ".join(a_tokens)
        if a_str and (a_str in label_token_str):
            score = 20 + len(a_str)
            if score > best_score:
                best_score = score
                best_match = canonical

    return best_match

ROOM_ALIASES: Dict[str, str] = {
    "bedroom": "bedroom", "master bedroom": "bedroom", "kids bedroom": "bedroom",
    "guest bedroom": "bedroom", "child room": "bedroom", "children room": "bedroom",
    "masterbed room": "bedroom", "masterbedroom": "bedroom",
    "secondbedroom": "bedroom", "second bedroom": "bedroom",
    "elderlyroom": "bedroom", "elderly room": "bedroom",
    "living room": "living_room", "livingroom": "living_room", "lounge": "living_room",
    "family room": "living_room", "sitting room": "living_room",
    "livingdiningroom": "living_room", "living dining room": "living_room",
    "kitchen": "kitchen", "kitchenette": "kitchen",
    "dining room": "dining_room", "diningroom": "dining_room",
    "office": "office", "study": "office", "home office": "office", "study room": "office",
    "bathroom": "bathroom", "restroom": "bathroom", "washroom": "bathroom",
    "laundry": "laundry_room", "laundry room": "laundry_room",
    "hallway": "hallway", "corridor": "hallway", "entrance": "hallway",
    "balcony": "balcony", "patio": "patio", "terrace": "patio",
    "garage": "garage", "storage room": "storage",
    "library": "library",
    "cloakroom": "hallway",
    "nannyroom": "bedroom", "nanny room": "bedroom",
    "bookstore": "bookstore", "book store": "bookstore",
    "buffet restaurant": "buffet_restaurant", "buffet_restaurant": "buffet_restaurant",
    "buffet": "buffet_restaurant",
    "children_room": "children_room", "childrens room": "children_room",
    "kids room": "children_room", "kid room": "children_room",
    "nursery": "children_room",
    "classroom": "classroom", "class room": "classroom",
    "computer room": "computer_room", "computer_room": "computer_room",
    "computer lab": "computer_room",
    "deli": "deli", "delicatessen": "deli",
    "florist": "florist_shop", "florist shop": "florist_shop",
    "florist_shop": "florist_shop", "flower shop": "florist_shop",
    "game room": "game_room", "game_room": "game_room",
    "gameroom": "game_room", "recreation room": "game_room",
}

def normalize_room(raw_label: str) -> Optional[str]:
    if not raw_label:
        return None
    label = raw_label.strip().lower()
    if not label:
        return None
    if label in ROOM_ALIASES:
        return ROOM_ALIASES[label]
    for alias, canonical in ROOM_ALIASES.items():
        if alias in label:
            return canonical
    return None

@dataclass
class ObjectObservation:
    category: str
    width: Optional[float] = None
    height: Optional[float] = None
    depth: Optional[float] = None
    z_position: Optional[float] = None
    z_min: Optional[float] = None
    room_type: Optional[str] = None
    support_surface: Optional[str] = None
    orientation_yaw: Optional[float] = None
    scene_id: Optional[str] = None
    source: str = ""
    x_pos: Optional[float] = None
    z_pos_horizontal: Optional[float] = None

@dataclass
class SceneRecord:
    scene_id: str
    room_type: Optional[str]
    object_categories: List[str] = field(default_factory=list)
    source: str = ""

@dataclass
class OrientationObservation:
    category: str
    scene_id: str
    angle_to_nearest_wall: Optional[float] = None
    angle_to_room_center: Optional[float] = None
    facing_pairs: List[Tuple[str, float, float]] = field(default_factory=list)

def _angle_diff(a: float, b: float) -> float:
    d = (a - b) % 360
    if d > 180:
        d = 360 - d
    return d

def _angle_to_point(from_x: float, from_z: float, to_x: float, to_z: float) -> float:
    dx = to_x - from_x
    dz = to_z - from_z
    return float(np.degrees(np.arctan2(dz, dx)) % 360)

def quaternion_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return 0.0
    x, y, z, w = (q / n)

    r00 = 1.0 - 2.0 * (y*y + z*z)
    r02 = 2.0 * (x*z + y*w)
    r20 = 2.0 * (x*z - y*w)
    r22 = 1.0 - 2.0 * (x*x + y*y)

    yaw = float(np.degrees(np.arctan2(r02, r22)) % 360.0)
    return yaw

def get_facing_direction(yaw_deg: float, category: str) -> float:
    offset = CATEGORY_FORWARD_OFFSET.get(category, 0.0)
    return float((yaw_deg + 90.0 + offset) % 360.0)

def _order_points_ccw(points_xz: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xz, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3:
        return pts
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    pts = pts[order]
    _, uniq_idx = np.unique(np.round(pts, 6), axis=0, return_index=True)
    pts = pts[np.sort(uniq_idx)]
    return pts

def polygon_centroid(vertices: np.ndarray) -> Tuple[float, float]:
    n = len(vertices)
    if n == 0:
        return 0.0, 0.0
    if n < 3:
        return float(np.mean(vertices[:, 0])), float(np.mean(vertices[:, 1]))
    cx, cz = 0.0, 0.0
    signed_area = 0.0
    for i in range(n):
        x0, z0 = vertices[i]
        x1, z1 = vertices[(i + 1) % n]
        cross = x0 * z1 - x1 * z0
        signed_area += cross
        cx += (x0 + x1) * cross
        cz += (z0 + z1) * cross
    signed_area *= 0.5
    if abs(signed_area) < 1e-10:
        return float(np.mean(vertices[:, 0])), float(np.mean(vertices[:, 1]))
    cx /= (6.0 * signed_area)
    cz /= (6.0 * signed_area)
    return float(cx), float(cz)

def polygon_to_wall_segments(vertices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    n = len(vertices)
    if n < 3:
        return []
    cx, cz = polygon_centroid(vertices)
    segments: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        dx = p2[0] - p1[0]
        dz = p2[1] - p1[1]
        edge_len = float(np.hypot(dx, dz))
        if edge_len < 1e-6:
            continue
        n1 = np.array([-dz, dx], dtype=np.float64) / edge_len
        n2 = np.array([dz, -dx], dtype=np.float64) / edge_len
        mid = (p1 + p2) / 2.0
        to_center = np.array([cx - mid[0], cz - mid[1]], dtype=np.float64)
        inward = n1 if float(np.dot(n1, to_center)) > 0 else n2
        normal_angle = float(np.degrees(np.arctan2(inward[1], inward[0])) % 360)
        segments.append((p1, p2, normal_angle))
    return segments

def point_to_segment_distance(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> float:
    abx, abz = bx - ax, bz - az
    apx, apz = px - ax, pz - az
    ab_sq = abx * abx + abz * abz
    if ab_sq < 1e-12:
        return float(np.hypot(apx, apz))
    t = max(0.0, min(1.0, (apx * abx + apz * abz) / ab_sq))
    proj_x = ax + t * abx
    proj_z = az + t * abz
    return float(np.hypot(px - proj_x, pz - proj_z))

def _mesh_centroid_xz(mesh_entry: dict) -> Optional[Tuple[float, float]]:
    xyz = mesh_entry.get("xyz", [])
    if not xyz or len(xyz) < 9:
        return None
    try:
        coords = np.array(xyz, dtype=np.float64).reshape(-1, 3)
        xz = coords[:, [0, 2]]
        c = xz.mean(axis=0)
        return float(c[0]), float(c[1])
    except Exception:
        return None

def extract_floor_polygon_from_mesh(
    mesh_entries: List[dict],
    room_children_uids: Set[str],
    *,
    room_hint_center_xz: Optional[Tuple[float, float]] = None,
) -> Optional[np.ndarray]:
    floor_meshes: List[dict] = []
    for mesh in mesh_entries:
        mesh_type = (mesh.get("type", "") or "").strip().lower()
        if "floor" in mesh_type:
            floor_meshes.append(mesh)
    if not floor_meshes:
        return None

    referenced = [m for m in floor_meshes if (m.get("uid", "") or "") in room_children_uids]
    chosen = referenced

    if not chosen and room_hint_center_xz is not None:
        hx, hz = room_hint_center_xz
        best = None
        best_dist = float("inf")
        for m in floor_meshes:
            c = _mesh_centroid_xz(m)
            if c is None:
                continue
            mx, mz = c
            d = (mx - hx) ** 2 + (mz - hz) ** 2
            if d < best_dist:
                best_dist = d
                best = m
        if best is not None:
            chosen = [best]

    if not chosen and len(floor_meshes) == 1:
        chosen = floor_meshes

    if not chosen:
        return None

    floor_points = []
    for mesh in chosen:
        xyz = mesh.get("xyz", [])
        if not xyz or len(xyz) < 9:
            continue
        try:
            coords = np.array(xyz, dtype=np.float64).reshape(-1, 3)
            floor_points.append(coords[:, [0, 2]])
        except Exception:
            continue

    if not floor_points:
        return None

    all_points = np.vstack(floor_points)
    if len(all_points) < 3:
        return None

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(all_points)
        poly = all_points[hull.vertices]
        return _order_points_ccw(poly)
    except Exception:
        unique_pts = np.unique(all_points, axis=0)
        if len(unique_pts) < 3:
            return None
        return _order_points_ccw(unique_pts)

def _infer_units_and_scale_dims(w: float, h: float, d: float) -> Tuple[float, float, float]:
    mx = max(w, h, d)
    if 0.01 <= mx <= 20.0:
        return w, h, d
    if 20.0 < mx <= 200.0:
        return w / 100.0, h / 100.0, d / 100.0
    return w / 1000.0, h / 1000.0, d / 1000.0

def _infer_units_and_scale_scalar(v: float) -> float:
    av = abs(v)
    if 20.0 < av <= 200.0:
        return v / 100.0
    if av > 200.0:
        return v / 1000.0
    return v

def load_model_info(model_info_path: str) -> Dict[str, dict]:
    info_map: Dict[str, dict] = {}
    path = Path(model_info_path)
    if not path.exists():
        log.warning(f"model_info.json not found at {path}")
        return info_map
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        mid = item.get("model_id")
        if mid:
            info_map[mid] = item
    log.info(f"Loaded model_info.json: {len(info_map)} models")
    return info_map

def read_ply_vertices(ply_path: Path) -> Optional[np.ndarray]:
    try:
        with open(ply_path, "rb") as f:
            header_lines: List[str] = []
            while True:
                line = f.readline()
                if not line:
                    return None
                line_str = line.decode("ascii", errors="ignore").strip()
                header_lines.append(line_str)
                if line_str == "end_header":
                    break

            fmt = "ascii"
            num_vertices = 0
            current_element = None
            vertex_props: List[Tuple[str, str]] = []

            for hl in header_lines:
                parts = hl.split()
                if not parts:
                    continue
                if parts[0] == "format":
                    fmt = parts[1]
                elif parts[0] == "element":
                    current_element = parts[1]
                    if parts[1] == "vertex":
                        num_vertices = int(parts[2])
                elif parts[0] == "property" and current_element == "vertex":
                    if parts[1] == "list":
                        vertex_props.append((parts[-1], "list"))
                    else:
                        vertex_props.append((parts[-1], parts[1]))

            if num_vertices == 0:
                return None

            prop_names = [p[0] for p in vertex_props]
            if "x" not in prop_names or "y" not in prop_names or "z" not in prop_names:
                return None
            x_idx = prop_names.index("x")
            y_idx = prop_names.index("y")
            z_idx = prop_names.index("z")

            if fmt == "ascii":
                vertices = np.zeros((num_vertices, 3), dtype=np.float64)
                for i in range(num_vertices):
                    line = f.readline().decode("ascii", errors="ignore").strip()
                    vals = line.split()
                    vertices[i, 0] = float(vals[x_idx])
                    vertices[i, 1] = float(vals[y_idx])
                    vertices[i, 2] = float(vals[z_idx])
                return vertices

            if fmt == "binary_little_endian":
                dtype_map = {
                    "float": ("f", 4), "float32": ("f", 4),
                    "double": ("d", 8), "float64": ("d", 8),
                    "uchar": ("B", 1), "uint8": ("B", 1),
                    "char": ("b", 1), "int8": ("b", 1),
                    "ushort": ("H", 2), "uint16": ("H", 2),
                    "short": ("h", 2), "int16": ("h", 2),
                    "uint": ("I", 4), "uint32": ("I", 4),
                    "int": ("i", 4), "int32": ("i", 4),
                }
                struct_fmt = "<"
                vertex_byte_size = 0
                for pname, ptype in vertex_props:
                    if ptype == "list":
                        return None
                    if ptype not in dtype_map:
                        return None
                    sfmt, sbytes = dtype_map[ptype]
                    struct_fmt += sfmt
                    vertex_byte_size += sbytes

                raw = f.read(num_vertices * vertex_byte_size)
                if len(raw) < num_vertices * vertex_byte_size:
                    return None

                vertices = np.zeros((num_vertices, 3), dtype=np.float64)
                for i in range(num_vertices):
                    offset = i * vertex_byte_size
                    vals = struct.unpack_from(struct_fmt, raw, offset)
                    vertices[i, 0] = float(vals[x_idx])
                    vertices[i, 1] = float(vals[y_idx])
                    vertices[i, 2] = float(vals[z_idx])
                return vertices

            return None
    except Exception:
        return None

class ThreeDFrontParser:
    def __init__(self, threedfront_dir: str, model_info_path: str | None = None):
        self.base_dir = Path(threedfront_dir)
        self.scene_dir = self._find_scene_dir()

        self.model_info: Dict[str, dict] = {}
        if model_info_path:
            self.model_info = load_model_info(model_info_path)
        else:
            for candidate in [
                self.base_dir.parent / "3D-FUTURE-model" / "model_info.json",
                self.base_dir.parent / "model_info.json",
                self.base_dir / "model_info.json",
            ]:
                if candidate.exists():
                    self.model_info = load_model_info(str(candidate))
                    break
            if not self.model_info:
                log.warning("model_info.json not found! Pass --model_info for better categories.")

    def _find_scene_dir(self) -> Path:
        candidates = [self.base_dir, self.base_dir / "3D-FRONT"]
        for c in candidates:
            jsons = list(c.glob("*.json"))
            if jsons:
                log.info(f"3D-FRONT scenes found at {c} ({len(jsons)} files)")
                return c
        log.warning(f"No 3D-FRONT scene JSONs found under {self.base_dir}")
        return self.base_dir

    def _get_category_for_furniture(self, furn: dict) -> Optional[str]:
        cat = furn.get("category", "")
        if isinstance(cat, str) and cat.strip():
            return cat.strip()
        title = furn.get("title", "")
        if isinstance(title, str) and title.strip():
            return title.strip()
        jid = furn.get("jid", "")
        if jid and jid in self.model_info:
            info = self.model_info[jid]
            for key in ["super-category", "category"]:
                val = info.get(key, "")
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    @staticmethod
    def _xz_aabb(x: float, z: float, w: Optional[float], d: Optional[float]) -> Optional[Tuple[float, float, float, float]]:
        if w is None or d is None:
            return None
        if w <= 1e-6 or d <= 1e-6:
            return None
        return (x - w / 2.0, x + w / 2.0, z - d / 2.0, z + d / 2.0)

    @staticmethod
    def _overlap_1d(a0: float, a1: float, b0: float, b1: float) -> float:
        return min(a1, b1) - max(a0, b0)

    def parse_all(self) -> Tuple[List[ObjectObservation], List[SceneRecord], List[OrientationObservation]]:
        observations: List[ObjectObservation] = []
        scenes: List[SceneRecord] = []
        orient_obs: List[OrientationObservation] = []

        json_files = sorted(self.scene_dir.glob("*.json"))
        log.info(f"Parsing {len(json_files)} 3D-FRONT scenes ...")

        error_count = 0
        geom_success = 0
        geom_fallback = 0

        for jf in json_files:
            try:
                obs, sc, oo, used_geom = self._parse_scene(jf)
                observations.extend(obs)
                scenes.extend(sc)
                orient_obs.extend(oo)
                if used_geom:
                    geom_success += 1
                else:
                    geom_fallback += 1
            except Exception as e:
                error_count += 1
                if error_count <= 10:
                    log.warning(f"Error parsing {jf.name}: {e}")
                elif error_count == 11:
                    log.warning("Suppressing further parse errors ...")

        if error_count:
            log.warning(f"3D-FRONT: {error_count}/{len(json_files)} scenes had parse errors")
        log.info(
            f"3D-FRONT: {len(observations)} observations, {len(scenes)} rooms, "
            f"{len(orient_obs)} orientation obs "
            f"(room geometry used: {geom_success}, heuristic fallback: {geom_fallback})"
        )
        return observations, scenes, orient_obs
    
    def _parse_scene(self, json_path: Path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        scene_uid = data.get("uid", json_path.stem)

        furniture_by_uid: Dict[str, dict] = {}
        for furn in data.get("furniture", []):
            uid = furn.get("uid")
            if uid:
                furniture_by_uid[uid] = furn

        mesh_entries = data.get("mesh", [])

        observations: List[ObjectObservation] = []
        scene_records: List[SceneRecord] = []
        orient_observations: List[OrientationObservation] = []
        any_used_geom = False

        scene_data = data.get("scene", {})
        if not isinstance(scene_data, dict):
            return observations, scene_records, orient_observations, False

        rooms = scene_data.get("room", [])

        for room_idx, room in enumerate(rooms):
            raw_room_type = room.get("type", "")
            room_type = normalize_room(raw_room_type)

            room_id = f"{scene_uid}_{raw_room_type}_{room_idx}"

            room_obj_cats: List[str] = []
            children = room.get("children", [])

            room_child_uids: Set[str] = set()
            for child in children:
                if isinstance(child, dict):
                    for key in ["ref", "instanceid", "uid"]:
                        val = child.get(key, "")
                        if val:
                            room_child_uids.add(val)

            hint_xz = None
            child_positions: List[Tuple[float, float]] = []
            for child in children:
                if not isinstance(child, dict):
                    continue
                pos = child.get("pos", None)
                if isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    try:
                        child_positions.append((float(pos[0]), float(pos[2])))
                    except Exception:
                        pass
            if child_positions:
                hx = float(np.mean([p[0] for p in child_positions]))
                hz = float(np.mean([p[1] for p in child_positions]))
                hint_xz = (hx, hz)

            floor_polygon = extract_floor_polygon_from_mesh(
                mesh_entries,
                room_child_uids,
                room_hint_center_xz=hint_xz,
            )

            wall_segments = None
            room_cx, room_cz = None, None
            used_room_geom = False

            if floor_polygon is not None and len(floor_polygon) >= 3:
                room_cx, room_cz = polygon_centroid(floor_polygon)
                wall_segments = polygon_to_wall_segments(floor_polygon)
                used_room_geom = True
                any_used_geom = True

            room_furniture: List[Tuple[dict, dict]] = []
            for child in children:
                if not isinstance(child, dict):
                    continue
                ref = child.get("ref", "")
                furn = furniture_by_uid.get(ref)
                if furn is None:
                    instance_id = child.get("instanceid", "")
                    furn = furniture_by_uid.get(instance_id)
                if furn is None:
                    continue
                if not furn.get("valid", True):
                    continue
                room_furniture.append((furn, child))

            def _get_y_pos(item: Tuple[dict, dict]) -> float:
                _furn, _child = item
                pos = _child.get("pos", [0, 0, 0])
                if isinstance(pos, (list, tuple)) and len(pos) > 1:
                    return float(pos[1])
                return 0.0

            room_furniture.sort(key=_get_y_pos)

            placed_objects: List[
                Tuple[str, Optional[float], Optional[float], float, float, Optional[float], Optional[float]]
            ] = []
            room_object_data: List[Tuple[str, float, float, float, Any]] = []

            for furn, child in room_furniture:
                raw_cat = self._get_category_for_furniture(furn)
                cat = normalize_category(raw_cat) if raw_cat else None
                if cat is None:
                    continue

                room_obj_cats.append(cat)

                pos = child.get("pos", [0, 0, 0])
                rot = child.get("rot", [0, 0, 0, 1])
                scale = child.get("scale", [1, 1, 1])

                if isinstance(pos, (list, tuple)):
                    pos = [float(p) for p in pos]
                else:
                    pos = [0.0, 0.0, 0.0]

                size = furn.get("size") or furn.get("bbox")
                w, h, d = None, None, None
                if isinstance(size, (list, tuple)) and len(size) >= 3:
                    if isinstance(size[0], (list, tuple)):
                        size = size[0]
                    w = abs(float(size[0]))
                    h = abs(float(size[1]))
                    d = abs(float(size[2]))
                    if isinstance(scale, (list, tuple)) and len(scale) >= 3:
                        w *= abs(float(scale[0]))
                        h *= abs(float(scale[1]))
                        d *= abs(float(scale[2]))
                    w, h, d = _infer_units_and_scale_dims(w, h, d)

                y_center = float(pos[1]) if len(pos) > 1 else None
                if y_center is not None:
                    y_center = _infer_units_and_scale_scalar(y_center)

                y_min = None
                if y_center is not None and h is not None:
                    y_min = y_center - h / 2.0

                support = None
                if y_min is not None:
                    tol_y = 0.05
                    tol_dist = 0.50
                    x0 = float(pos[0]) if len(pos) > 0 else 0.0
                    z0 = float(pos[2]) if len(pos) > 2 else 0.0
                    cur_aabb = self._xz_aabb(x0, z0, w, d)

                    best_cat = None
                    best_score = -1e9

                    for prev_cat, prev_ymin, prev_ymax, px, pz, prev_w, prev_d in placed_objects:
                        if prev_ymax is None:
                            continue
                        if abs(prev_ymax - y_min) > tol_y:
                            continue

                        score = -1e9
                        if cur_aabb is not None:
                            prev_aabb = self._xz_aabb(px, pz, prev_w, prev_d)
                            if prev_aabb is not None:
                                xo = self._overlap_1d(cur_aabb[0], cur_aabb[1], prev_aabb[0], prev_aabb[1])
                                zo = self._overlap_1d(cur_aabb[2], cur_aabb[3], prev_aabb[2], prev_aabb[3])
                                if xo > 0 and zo > 0:
                                    score = float(xo * zo)
                            else:
                                dist = float(np.hypot(x0 - px, z0 - pz))
                                if dist <= tol_dist:
                                    score = 1.0 - (dist / tol_dist)
                        else:
                            dist = float(np.hypot(x0 - px, z0 - pz))
                            if dist <= tol_dist:
                                score = 1.0 - (dist / tol_dist)

                        if score > best_score:
                            best_score = score
                            best_cat = prev_cat

                    if best_cat is not None and best_score > 0:
                        support = best_cat
                    elif y_min < 0.05:
                        support = "floor"

                y_max = (y_center + h / 2.0) if (y_center is not None and h is not None) else None
                placed_objects.append((
                    cat,
                    y_min,
                    y_max,
                    float(pos[0]) if len(pos) > 0 else 0.0,
                    float(pos[2]) if len(pos) > 2 else 0.0,
                    w,
                    d,
                ))

                yaw = None
                if isinstance(rot, (list, tuple)):
                    rot_f = [float(r) for r in rot]
                    if len(rot_f) == 4:
                        qx, qy, qz, qw = rot_f
                        yaw = quaternion_to_yaw(qx, qy, qz, qw)
                    elif len(rot_f) == 3:
                        yaw = float(rot_f[1])

                x_horizontal = float(pos[0]) if len(pos) > 0 else None
                z_horizontal = float(pos[2]) if len(pos) > 2 else None

                observations.append(ObjectObservation(
                    category=cat,
                    width=w,
                    height=h,
                    depth=d,
                    z_position=y_center,
                    z_min=y_min,
                    room_type=room_type,
                    support_surface=support,
                    orientation_yaw=yaw,
                    scene_id=room_id,
                    source="3dfront",
                    x_pos=x_horizontal,
                    z_pos_horizontal=z_horizontal,
                ))

                if yaw is not None and x_horizontal is not None and z_horizontal is not None:
                    room_object_data.append((cat, yaw, x_horizontal, z_horizontal, None))

            if len(room_object_data) >= 2:
                oo = self._mine_room_orientation(
                    room_object_data,
                    room_id,
                    wall_segments=wall_segments,
                    room_center=(room_cx, room_cz),
                    floor_polygon=floor_polygon,
                    used_room_geom=used_room_geom,
                )
                orient_observations.extend(oo)

            if room_obj_cats:
                scene_records.append(SceneRecord(
                    scene_id=room_id,
                    room_type=room_type,
                    object_categories=room_obj_cats,
                    source="3dfront",
                ))

        return observations, scene_records, orient_observations, any_used_geom

    def _mine_room_orientation(
        self,
        room_objects: List[Tuple[str, float, float, float, Any]],
        scene_id: str,
        wall_segments: Optional[List] = None,
        room_center: Optional[Tuple[float, float]] = None,
        floor_polygon: Optional[np.ndarray] = None,
        used_room_geom: bool = False,
    ) -> List[OrientationObservation]:
        orient_obs: List[OrientationObservation] = []
        xs = [o[2] for o in room_objects]
        zs = [o[3] for o in room_objects]

        if room_center and room_center[0] is not None:
            cx, cz = room_center
        else:
            cx = (min(xs) + max(xs)) / 2.0
            cz = (min(zs) + max(zs)) / 2.0

        if wall_segments and len(wall_segments) >= 3:
            walls = wall_segments
        elif floor_polygon is not None and len(floor_polygon) >= 3:
            walls = polygon_to_wall_segments(floor_polygon)
        else:
            margin = 0.5
            corners = np.array(
                [
                    [min(xs) - margin, min(zs) - margin],
                    [max(xs) + margin, min(zs) - margin],
                    [max(xs) + margin, max(zs) + margin],
                    [min(xs) - margin, max(zs) + margin],
                ],
                dtype=np.float64,
            )
            walls = polygon_to_wall_segments(corners)

        for i, (cat, yaw, ox, oz, _) in enumerate(room_objects):
            facing_dir = get_facing_direction(yaw, cat)

            angle_to_wall = None
            if cat in BACK_TO_WALL_CATEGORIES and walls:
                best_wall_dist = float("inf")
                best_wall_normal = None
                for p1, p2, normal_angle in walls:
                    dist = point_to_segment_distance(ox, oz, float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]))
                    if dist < best_wall_dist:
                        best_wall_dist = dist
                        best_wall_normal = normal_angle
                if best_wall_normal is not None:
                    back_dir = (facing_dir + 180.0) % 360.0
                    angle_to_wall = _angle_diff(back_dir, best_wall_normal)

            angle_to_center = None
            if cat in FACES_CENTER_CATEGORIES:
                dist_to_center = float(np.hypot(cx - ox, cz - oz))
                if dist_to_center > 0.1:
                    center_direction = _angle_to_point(ox, oz, cx, cz)
                    angle_to_center = _angle_diff(facing_dir, center_direction)

            facing_pairs: List[Tuple[str, float, float]] = []
            if cat in FACES_PAIR_CATEGORIES:
                for j, (other_cat, _, other_x, other_z, _) in enumerate(room_objects):
                    if i == j:
                        continue
                    dist = float(np.hypot(ox - other_x, oz - other_z))
                    if dist > 5.0:
                        continue
                    direction_to_other = _angle_to_point(ox, oz, other_x, other_z)
                    angle_facing_other = _angle_diff(facing_dir, direction_to_other)
                    facing_pairs.append((other_cat, angle_facing_other, dist))

            if angle_to_wall is not None or angle_to_center is not None or facing_pairs:
                orient_obs.append(
                    OrientationObservation(
                        category=cat,
                        scene_id=scene_id,
                        angle_to_nearest_wall=angle_to_wall,
                        angle_to_room_center=angle_to_center,
                        facing_pairs=facing_pairs,
                    )
                )
        return orient_obs

def infer_room_type_from_objects(object_cats: List[str]) -> Optional[str]:
    if not object_cats:
        return None
    cat_set = set(object_cats)
    room_indicators = {
        "bedroom": {"bed", "nightstand", "dresser", "wardrobe", "pillow", "blanket"},
        "living_room": {"sofa", "coffee_table", "tv", "armchair", "tv_stand", "rug"},
        "kitchen": {"refrigerator", "oven", "stove", "microwave", "dishwasher", "sink"},
        "bathroom": {"toilet", "bathtub", "shower", "sink", "mirror"},
        "office": {"desk", "office_chair", "monitor", "computer", "filing_cabinet", "printer"},
        "dining_room": {"dining_table", "chair"},
    }
    scores = {room: len(cat_set & ind) for room, ind in room_indicators.items()}
    if not scores or max(scores.values()) == 0:
        return None
    return max(scores, key=scores.get)

class ScanNetParser:
    SCANNET_LABELS = {
        1: "wall", 2: "floor", 3: "cabinet", 4: "bed", 5: "chair",
        6: "sofa", 7: "table", 8: "door", 9: "window", 10: "bookshelf",
        11: "picture", 12: "counter", 13: "desk", 14: "curtain",
        15: "refrigerator", 16: "shower curtain", 17: "toilet", 18: "sink",
        19: "bathtub", 20: "otherfurniture", 22: "dresser",
        23: "nightstand", 24: "lamp", 25: "wardrobe", 26: "mirror",
        27: "tv", 28: "pillow", 29: "plant", 30: "monitor",
        31: "rug", 32: "shelving", 33: "stool", 34: "oven",
        35: "microwave", 36: "fan", 37: "bench", 38: "filing cabinet",
        39: "dishwasher", 40: "washing machine",
    }

    def __init__(self, scannet_dir: str):
        self.base_dir = Path(scannet_dir)
        self.scans_dir = self._find_scans_dir()
        self.scene_types = self._load_scene_types()
        self.label_map = self._load_label_map()

    def _find_scans_dir(self) -> Path:
        candidates = [self.base_dir / "scans", self.base_dir / "ScanNet" / "scans", self.base_dir]
        for c in candidates:
            if c.is_dir() and any(c.iterdir()):
                return c
        return self.base_dir / "scans"

    def _load_scene_types(self) -> Dict[str, str]:
        scene_types: Dict[str, str] = {}
        for scene_dir in sorted(self.scans_dir.glob("scene*")):
            txt_file = scene_dir / f"{scene_dir.name}.txt"
            if txt_file.exists():
                try:
                    with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if "sceneType" in line:
                                parts = line.strip().split("=")
                                if len(parts) == 2:
                                    room = normalize_room(parts[1].strip())
                                    if room:
                                        scene_types[scene_dir.name] = room
                except Exception:
                    pass
        log.info(f"Loaded room types for {len(scene_types)} ScanNet scenes")
        return scene_types

    def _load_label_map(self) -> Dict[int, str]:
        tsv_path = self.base_dir / "scannetv2-labels.combined.tsv"
        if tsv_path.exists():
            mapping: Dict[int, str] = {}
            with open(tsv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    try:
                        nyu_id = int(row.get("nyu40id", -1))
                        label = (row.get("nyu40class", "") or "").strip()
                        if nyu_id > 0 and label:
                            mapping[nyu_id] = label
                    except Exception:
                        pass
            if mapping:
                log.info(f"Loaded {len(mapping)} labels from TSV")
                return mapping
        return self.SCANNET_LABELS

    def _load_axis_alignment(self, scene_dir: Path, scene_id: str) -> Optional[np.ndarray]:
        txt_file = scene_dir / f"{scene_id}.txt"
        if not txt_file.exists():
            return None
        try:
            with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "axisAlignment" in line:
                        parts = line.strip().split("=")
                        if len(parts) == 2:
                            vals = [float(x) for x in parts[1].strip().split()]
                            if len(vals) == 16:
                                return np.array(vals, dtype=np.float64).reshape(4, 4)
        except Exception:
            pass
        return None

    def _load_segmentation(self, scene_dir: Path, scene_id: str) -> Optional[List[int]]:
        seg_file = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
        if not seg_file.exists():
            seg_file = scene_dir / f"{scene_id}_vh_clean.segs.json"
        if not seg_file.exists():
            return None
        try:
            with open(seg_file, "r", encoding="utf-8", errors="ignore") as f:
                seg_data = json.load(f)
            return seg_data.get("segIndices", []) or None
        except Exception:
            return None

    def _load_aggregation(self, scene_dir: Path, scene_id: str) -> Optional[List[dict]]:
        for agg_name in [f"{scene_id}_vh_clean.aggregation.json", f"{scene_id}.aggregation.json"]:
            agg_file = scene_dir / agg_name
            if agg_file.exists():
                try:
                    with open(agg_file, "r", encoding="utf-8", errors="ignore") as f:
                        agg_data = json.load(f)
                    seg_groups = agg_data.get("segGroups", [])
                    if seg_groups:
                        return seg_groups
                except Exception:
                    pass
        return None

    def parse_all(self) -> Tuple[List[ObjectObservation], List[SceneRecord]]:
        observations: List[ObjectObservation] = []
        scenes: List[SceneRecord] = []
        scene_dirs = sorted(self.scans_dir.glob("scene*"))
        log.info(f"Parsing {len(scene_dirs)} ScanNet scenes ...")

        bbox_success = 0
        agg_only = 0
        failed = 0

        for sd in scene_dirs:
            try:
                obs, sc, had_bbox = self._parse_scene(sd)
                observations.extend(obs)
                if sc:
                    scenes.append(sc)
                if had_bbox:
                    bbox_success += 1
                elif obs:
                    agg_only += 1
            except Exception:
                failed += 1

        log.info(
            f"ScanNet: {len(observations)} observations, {len(scenes)} scenes "
            f"({bbox_success} bbox, {agg_only} agg-only, {failed} failed)"
        )
        return observations, scenes

    def _parse_scene(self, scene_dir: Path):
        scene_id = scene_dir.name
        room_type = self.scene_types.get(scene_id)
        observations: List[ObjectObservation] = []
        object_cats: List[str] = []

        bbox_file = scene_dir / f"{scene_id}_aligned_bbox.npy"
        if not bbox_file.exists():
            bbox_file = scene_dir / f"{scene_id}_bbox.npy"

        if bbox_file.exists():
            bboxes = np.load(str(bbox_file))
            for row in bboxes:
                if len(row) < 7:
                    continue
                cx, cy, cz, dx, dy, dz = row[:6]
                label_id = int(row[6])
                raw_label = self.label_map.get(label_id, "")
                cat = normalize_category(raw_label)
                if cat is None:
                    continue
                object_cats.append(cat)

                w = abs(float(dx))
                d = abs(float(dy))
                h = abs(float(dz))

                z_pos = float(cz)
                z_min = z_pos - h / 2.0
                support = "floor" if z_min < 0.05 else None

                observations.append(ObjectObservation(
                    category=cat, width=w, height=h, depth=d,
                    z_position=z_pos, z_min=z_min,
                    room_type=room_type, support_surface=support,
                    scene_id=scene_id, source="scannet",
                ))

            if room_type is None:
                room_type = infer_room_type_from_objects(object_cats)
                if room_type is not None:
                    for o in observations:
                        if o.room_type is None:
                            o.room_type = room_type

            sc = SceneRecord(
                scene_id=scene_id, room_type=room_type, object_categories=object_cats, source="scannet"
            ) if object_cats else None
            return observations, sc, True

        seg_groups = self._load_aggregation(scene_dir, scene_id)
        seg_indices = self._load_segmentation(scene_dir, scene_id)

        vertices = None
        for ply_name in [f"{scene_id}_vh_clean_2.ply", f"{scene_id}_vh_clean.ply"]:
            ply_path = scene_dir / ply_name
            if ply_path.exists():
                vertices = read_ply_vertices(ply_path)
                if vertices is not None:
                    break

        if vertices is not None:
            alignment = self._load_axis_alignment(scene_dir, scene_id)
            if alignment is not None:
                n = vertices.shape[0]
                ones = np.ones((n, 1), dtype=np.float64)
                pts_homo = np.hstack([vertices, ones])
                pts_aligned = (alignment @ pts_homo.T).T[:, :3]
                vertices = pts_aligned

        if seg_groups and seg_indices is not None and vertices is not None:
            seg_to_verts: Dict[int, List[int]] = defaultdict(list)
            for vert_idx, seg_id in enumerate(seg_indices):
                if vert_idx < len(vertices):
                    seg_to_verts[int(seg_id)].append(vert_idx)

            bbox_observations: List[Tuple[ObjectObservation, np.ndarray, np.ndarray]] = []

            for sg in seg_groups:
                raw_label = sg.get("label", "")
                cat = normalize_category(raw_label)
                if cat is None:
                    continue

                obj_vert_indices: List[int] = []
                for seg_id in sg.get("segments", []):
                    obj_vert_indices.extend(seg_to_verts.get(int(seg_id), []))

                if len(obj_vert_indices) < 3:
                    object_cats.append(cat)
                    observations.append(ObjectObservation(category=cat, room_type=room_type, scene_id=scene_id, source="scannet"))
                    continue

                obj_verts = vertices[obj_vert_indices]
                bbox_min = obj_verts.min(axis=0)
                bbox_max = obj_verts.max(axis=0)
                bbox_center = (bbox_min + bbox_max) / 2.0
                bbox_extent = bbox_max - bbox_min

                w = float(bbox_extent[0])
                d = float(bbox_extent[1])
                h = float(bbox_extent[2])
                z_pos = float(bbox_center[2])
                z_min_val = float(bbox_min[2])

                if w < 0.005 or d < 0.005 or h < 0.005:
                    object_cats.append(cat)
                    observations.append(ObjectObservation(category=cat, room_type=room_type, scene_id=scene_id, source="scannet"))
                    continue

                object_cats.append(cat)
                obs = ObjectObservation(
                    category=cat, width=w, height=h, depth=d,
                    z_position=z_pos, z_min=z_min_val,
                    room_type=room_type, support_surface=None,
                    scene_id=scene_id, source="scannet",
                )
                observations.append(obs)
                bbox_observations.append((obs, bbox_min.copy(), bbox_max.copy()))

            for i, (obs_i, bmin_i, bmax_i) in enumerate(bbox_observations):
                z_min_i = float(bmin_i[2])
                if z_min_i < 0.05:
                    obs_i.support_surface = "floor"
                    continue

                tolerance = 0.10
                best_support = None
                best_dist = tolerance

                for j, (obs_j, bmin_j, bmax_j) in enumerate(bbox_observations):
                    if i == j:
                        continue
                    other_top = float(bmax_j[2])
                    dist = abs(other_top - z_min_i)

                    if dist < best_dist and other_top <= z_min_i + tolerance:
                        x_overlap = min(float(bmax_i[0]), float(bmax_j[0])) - max(float(bmin_i[0]), float(bmin_j[0]))
                        y_overlap = min(float(bmax_i[1]), float(bmax_j[1])) - max(float(bmin_i[1]), float(bmin_j[1]))
                        if x_overlap > 0 and y_overlap > 0:
                            best_support = obs_j.category
                            best_dist = dist

                if best_support:
                    obs_i.support_surface = best_support

            if room_type is None:
                room_type = infer_room_type_from_objects(object_cats)
                if room_type is not None:
                    for o in observations:
                        if o.room_type is None:
                            o.room_type = room_type

            sc = SceneRecord(scene_id=scene_id, room_type=room_type, object_categories=object_cats, source="scannet") if object_cats else None
            return observations, sc, len(bbox_observations) > 0

        if seg_groups:
            for sg in seg_groups:
                raw_label = sg.get("label", "")
                cat = normalize_category(raw_label)
                if cat is None:
                    continue
                object_cats.append(cat)
                observations.append(ObjectObservation(category=cat, room_type=room_type, scene_id=scene_id, source="scannet"))

            if room_type is None:
                room_type = infer_room_type_from_objects(object_cats)
                if room_type is not None:
                    for o in observations:
                        if o.room_type is None:
                            o.room_type = room_type

            sc = SceneRecord(scene_id=scene_id, room_type=room_type, object_categories=object_cats, source="scannet") if object_cats else None
            return observations, sc, False

        return [], None, False

class VisualGenomeParser:
    SUPPORT_PREDICATES = {
        "on", "on top of", "sitting on", "standing on",
        "resting on", "placed on", "mounted on", "lying on",
        "on top", "atop",
    }
    PLAUSIBLE_SUPPORT_SURFACES = {
        "floor", "table", "desk", "nightstand", "side_table", "coffee_table",
        "dining_table", "dressing_table", "console_table", "shelf", "wall_shelf",
        "bookshelf", "cabinet", "dresser", "tv_stand", "sideboard", "counter",
        "bed", "sofa", "chair", "armchair", "bench", "ottoman", "stool",
        "rug", "window", "fireplace", "refrigerator",
    }
    IMPLAUSIBLE_SUPPORT_SURFACES = {
        "painting", "curtain", "pillow", "blanket", "photo_frame",
        "vase", "plant", "clock", "lamp", "table_lamp", "floor_lamp",
        "ceiling_lamp", "pendant_lamp", "wall_lamp", "chandelier",
        "speaker", "laptop", "keyboard", "monitor",
    }
    TYPICAL_HEIGHT_RANGE_M = {
        "rug": (0.0, 0.01), "ottoman": (0.0, 0.22), "stool": (0.0, 0.35),
        "bench": (0.0, 0.25), "chair": (0.0, 0.45), "armchair": (0.0, 0.45),
        "office_chair": (0.0, 0.50), "coffee_table": (0.0, 0.25),
        "side_table": (0.0, 0.35), "nightstand": (0.0, 0.35),
        "bed": (0.0, 0.35), "sofa": (0.0, 0.40), "recliner": (0.0, 0.45),
        "desk": (0.0, 0.38), "dining_table": (0.0, 0.38), "table": (0.0, 0.38),
        "tv_stand": (0.0, 0.30), "dresser": (0.0, 0.50),
        "bookshelf": (0.0, 0.90), "wardrobe": (0.0, 1.00),
        "cabinet": (0.0, 0.60), "sideboard": (0.0, 0.45),
        "console_table": (0.0, 0.40), "floor_lamp": (0.0, 0.80),
        "plant": (0.0, 0.40), "fireplace": (0.0, 0.55),
        "table_lamp": (0.70, 0.90), "lamp": (0.60, 0.80),
        "vase": (0.70, 0.85), "photo_frame": (0.70, 0.85),
        "clock": (0.70, 0.85), "monitor": (0.70, 0.95),
        "tv": (0.50, 0.85), "laptop": (0.70, 0.80),
        "computer": (0.60, 0.80), "speaker": (0.50, 0.70),
        "microwave": (0.85, 1.05), "printer": (0.70, 0.85),
        "painting": (1.20, 1.60), "mirror": (0.80, 1.30),
        "wall_shelf": (1.00, 1.30), "shelf": (0.40, 0.80),
        "wall_lamp": (1.50, 1.80), "curtain": (1.50, 2.00),
        "ceiling_lamp": (2.20, 2.40), "chandelier": (2.00, 2.30),
        "pendant_lamp": (1.80, 2.20), "ceiling_fan": (2.20, 2.40),
        "refrigerator": (0.0, 0.85), "oven": (0.0, 0.45),
        "stove": (0.0, 0.45), "dishwasher": (0.0, 0.45),
        "washing_machine": (0.0, 0.45), "toilet": (0.0, 0.25),
        "sink": (0.75, 0.90), "bathtub": (0.0, 0.30),
        "pillow": (0.30, 0.50), "blanket": (0.25, 0.40),
    }
    _WEIGHT_TIER = {
        "pillow": 0, "blanket": 0, "vase": 0, "photo_frame": 0,
        "clock": 0, "lamp": 0, "table_lamp": 0,
        "chair": 1, "stool": 1, "nightstand": 1, "side_table": 1,
        "monitor": 1, "tv": 1, "speaker": 1, "plant": 1,
        "ottoman": 1, "coffee_table": 1, "bench": 1, "mirror": 1,
        "desk": 2, "table": 2, "dining_table": 2, "sofa": 2,
        "armchair": 2, "dresser": 2, "tv_stand": 2, "cabinet": 2,
        "sideboard": 2, "console_table": 2, "bookshelf": 2, "shelf": 2,
        "bed": 3, "wardrobe": 3, "refrigerator": 3, "bathtub": 3,
    }

    def __init__(self, vg_dir: str):
        self.base_dir = Path(vg_dir)

    def _load_image_dimensions(self) -> Dict[int, Tuple[int, int]]:
        img_dims: Dict[int, Tuple[int, int]] = {}
        img_data_file = self.base_dir / "image_data.json"
        if not img_data_file.exists():
            return img_dims
        log.info("Loading VG image_data.json ...")
        with open(img_data_file, "r", encoding="utf-8", errors="ignore") as f:
            img_data_list = json.load(f)
        for item in img_data_list:
            img_id = item.get("image_id")
            w = item.get("width", 0)
            h = item.get("height", 0)
            if img_id and w > 0 and h > 0:
                img_dims[int(img_id)] = (int(w), int(h))
        log.info(f"Loaded dimensions for {len(img_dims)} VG images")
        return img_dims

    def _load_scene_graph_attributes(self) -> Dict[int, Dict[int, List[str]]]:
        attrs: Dict[int, Dict[int, List[str]]] = {}
        sg_file = self.base_dir / "scene_graphs.json"
        if not sg_file.exists():
            log.info("scene_graphs.json not found \u2014 skipping attribute extraction")
            return attrs
        log.info("Loading Visual Genome scene_graphs.json ...")
        with open(sg_file, "r", encoding="utf-8", errors="ignore") as f:
            scene_graphs = json.load(f)
        for sg in scene_graphs:
            img_id = sg.get("image_id")
            if img_id is None:
                continue
            obj_attrs: Dict[int, List[str]] = {}
            for obj in sg.get("objects", []):
                obj_id = obj.get("object_id")
                obj_attr_list = obj.get("attributes", [])
                if obj_id and obj_attr_list:
                    obj_attrs[obj_id] = [
                        a.lower().strip() for a in obj_attr_list if isinstance(a, str)
                    ]
            if obj_attrs:
                attrs[int(img_id)] = obj_attrs
        log.info(f"Loaded attributes for {len(attrs)} VG images")
        return attrs

    def parse_all(self) -> Tuple[List[ObjectObservation], List[SceneRecord]]:
        observations: List[ObjectObservation] = []
        scenes: List[SceneRecord] = []

        objects_file = self.base_dir / "objects.json"
        relationships_file = self.base_dir / "relationships.json"
        if not objects_file.exists():
            log.warning(f"objects.json not found at {objects_file}")
            return observations, scenes

        image_dims = self._load_image_dimensions()

        log.info("Loading VG objects.json ...")
        with open(objects_file, "r", encoding="utf-8", errors="ignore") as f:
            vg_objects = json.load(f)

        image_objects: Dict[int, List[str]] = {}

        for img_data in vg_objects:
            img_id = img_data.get("image_id")
            if img_id is None:
                continue
            img_id = int(img_id)
            img_w, img_h = image_dims.get(img_id, (0, 0))

            cats: List[str] = []
            img_observations: List[Tuple[str, Optional[float], Optional[float]]] = []

            for obj in img_data.get("objects", []):
                names = obj.get("names", [])
                synsets = obj.get("synsets", [])
                raw_name = names[0] if names else ""
                cat = normalize_category(raw_name)
                if cat is None and synsets:
                    for syn in synsets:
                        cat_try = normalize_category(syn.split(".")[0])
                        if cat_try:
                            cat = cat_try
                            break
                if cat is None:
                    continue

                cats.append(cat)

                z_position: Optional[float] = None
                z_min: Optional[float] = None

                if cat in self.TYPICAL_HEIGHT_RANGE_M:
                    z_bot, z_ctr = self.TYPICAL_HEIGHT_RANGE_M[cat]
                    z_min = z_bot
                    z_position = z_ctr
                elif img_h > 0:
                    obj_y = obj.get("y", 0)
                    obj_h_px = obj.get("h", 0)
                    if obj_h_px > 0:
                        ROOM_HEIGHT = 2.5
                        norm_y_bottom = (obj_y + obj_h_px) / img_h
                        norm_y_top = obj_y / img_h
                        z_min = ROOM_HEIGHT * (1.0 - norm_y_bottom)
                        z_top = ROOM_HEIGHT * (1.0 - norm_y_top)
                        z_position = (z_min + z_top) / 2.0
                        z_min = max(0.0, min(float(z_min), ROOM_HEIGHT))
                        z_position = max(0.0, min(float(z_position), ROOM_HEIGHT))

                img_observations.append((cat, z_position, z_min))

            if cats:
                image_objects[img_id] = cats
                room_type = infer_room_type_from_objects(cats)
                scenes.append(SceneRecord(
                    scene_id=f"vg_{img_id}", room_type=room_type,
                    object_categories=cats, source="vg",
                ))
                for cat, z_position, z_min in img_observations:
                    observations.append(ObjectObservation(
                        category=cat, room_type=room_type,
                        z_position=z_position, z_min=z_min,
                        scene_id=f"vg_{img_id}", source="vg",
                    ))

        if relationships_file.exists():
            log.info("Loading VG relationships.json ...")
            with open(relationships_file, "r", encoding="utf-8", errors="ignore") as f:
                vg_rels = json.load(f)
            observations.extend(self._parse_relationships(vg_rels, image_objects))

        log.info(f"Visual Genome: {len(observations)} observations, {len(scenes)} scenes")
        return observations, scenes
    
    def _parse_relationships(self, vg_rels, image_objects: Dict[int, List[str]]) -> List[ObjectObservation]:
        support_observations: List[ObjectObservation] = []
        filtered_count = 0

        for img_data in vg_rels:
            img_id = img_data.get("image_id")
            if img_id is None:
                continue
            img_id = int(img_id)

            for rel in img_data.get("relationships", []):
                predicate = (rel.get("predicate", "") or "").lower().strip()
                if predicate not in self.SUPPORT_PREDICATES:
                    continue

                subj = rel.get("subject", {}) or {}
                obj = rel.get("object", {}) or {}
                subj_names = subj.get("names", [subj.get("name", "")])
                obj_names = obj.get("names", [obj.get("name", "")])

                subj_cat = normalize_category(subj_names[0] if subj_names else "")
                obj_cat = normalize_category(obj_names[0] if obj_names else "")
                if subj_cat is None or obj_cat is None:
                    continue

                if obj_cat in self.IMPLAUSIBLE_SUPPORT_SURFACES:
                    filtered_count += 1
                    continue
                if subj_cat == obj_cat and subj_cat not in {"shelf", "stool", "bench"}:
                    filtered_count += 1
                    continue
                if self._is_heavier_than(subj_cat, obj_cat):
                    filtered_count += 1
                    continue

                subj_y = subj.get("y", 0) + subj.get("h", 0)
                obj_bottom = obj.get("y", 0) + obj.get("h", 0)
                if subj_y > 0 and obj_bottom > 0 and subj_y > obj_bottom + 50:
                    filtered_count += 1
                    continue

                room_type = None
                if img_id in image_objects:
                    room_type = infer_room_type_from_objects(image_objects[img_id])

                support_observations.append(ObjectObservation(
                    category=subj_cat,
                    support_surface=obj_cat,
                    room_type=room_type,
                    scene_id=f"vg_{img_id}",
                    source="vg",
                ))

        if filtered_count > 0:
            log.info(f"VG relationships: {len(support_observations)} kept, {filtered_count} filtered")
        return support_observations

    def _is_heavier_than(self, subj_cat: str, obj_cat: str) -> bool:
        return self._WEIGHT_TIER.get(subj_cat, 1) >= self._WEIGHT_TIER.get(obj_cat, 1) + 2

class OntologyBuilder:
    def __init__(self):
        self.observations: List[ObjectObservation] = []
        self.scenes: List[SceneRecord] = []
        self.orientation_observations: List[OrientationObservation] = []

    def add(self, obs: List[ObjectObservation], scenes: List[SceneRecord], orient_obs: Optional[List[OrientationObservation]] = None):
        self.observations.extend(obs)
        self.scenes.extend(scenes)
        if orient_obs:
            self.orientation_observations.extend(orient_obs)

    def build(self) -> Dict[str, dict]:
        log.info(
            f"Building ontology from {len(self.observations)} observations, "
            f"{len(self.scenes)} scenes, "
            f"{len(self.orientation_observations)} orientation observations"
        )

        ontology: Dict[str, dict] = {}
        by_cat: Dict[str, List[ObjectObservation]] = defaultdict(list)
        for obs in self.observations:
            by_cat[obs.category].append(obs)

        for cat, obs_list in sorted(by_cat.items()):
            ontology[cat] = self._build_category_entry(cat, obs_list)

        global_stats, per_room_stats = self._compute_cooccurrence_normalized()

        for cat in ontology:
            if cat in global_stats:
                top = sorted(global_stats[cat].items(), key=lambda kv: -(kv[1].get("count", 0)))[:50]
                ontology[cat]["cooccurrence"] = {k: v for k, v in top if k != cat}

            if cat in per_room_stats:
                per_room_out = {}
                for room, stats in per_room_stats[cat].items():
                    top_r = sorted(stats.items(), key=lambda kv: -(kv[1].get("count", 0)))[:20]
                    per_room_out[room] = {k: v for k, v in top_r if k != cat}
                if per_room_out:
                    ontology[cat]["cooccurrence_by_room"] = per_room_out

        orient_data = self._aggregate_orientation_relationships()
        for cat, orient_entry in orient_data.items():
            if cat in ontology:
                ontology[cat]["orientation_relationships"] = orient_entry

        return ontology

    def _build_category_entry(self, cat: str, obs_list: List[ObjectObservation]) -> dict:
        entry: dict = {
            "category": cat,
            "count": len(obs_list),
            "sources": sorted(list(set(o.source for o in obs_list))),
        }

        widths = [o.width for o in obs_list if o.width is not None and 0.01 < o.width < 10]
        heights = [o.height for o in obs_list if o.height is not None and 0.01 < o.height < 10]
        depths = [o.depth for o in obs_list if o.depth is not None and 0.01 < o.depth < 10]

        if widths:
            entry["dimensions"] = {
                "width_m": self._percentile_stats(widths),
                "height_m": self._percentile_stats(heights) if heights else None,
                "depth_m": self._percentile_stats(depths) if depths else None,
                "n_width": len(widths),
                "n_height": len(heights),
                "n_depth": len(depths),
            }

        room_counts = Counter(o.room_type for o in obs_list if o.room_type is not None)
        if room_counts:
            total = sum(room_counts.values())
            entry["room_associations"] = {
                room: {"count": int(cnt), "fraction": round(cnt / total, 3)}
                for room, cnt in room_counts.most_common()
            }

        support_counts = Counter(o.support_surface for o in obs_list if o.support_surface is not None)
        if support_counts:
            total_support = sum(support_counts.values())
            entry["support_surfaces"] = {
                surf: {"count": int(cnt), "fraction": round(cnt / total_support, 3)}
                for surf, cnt in support_counts.most_common()
            }

        z3 = [o.z_position for o in obs_list
              if o.source in {"3dfront", "scannet"} and o.z_position is not None and -1 < o.z_position < 5]
        z3_min = [o.z_min for o in obs_list
                  if o.source in {"3dfront", "scannet"} and o.z_min is not None and -1 < o.z_min < 5]
        if z3:
            entry["height_band_3d"] = {
                "z_center_m": self._percentile_stats(z3),
                "z_bottom_m": self._percentile_stats(z3_min) if z3_min else None,
            }

        zvg = [o.z_position for o in obs_list
               if o.source == "vg" and o.z_position is not None and -1 < o.z_position < 5]
        zvg_min = [o.z_min for o in obs_list
                   if o.source == "vg" and o.z_min is not None and -1 < o.z_min < 5]
        if zvg:
            entry["height_band_vg"] = {
                "z_center_m": self._percentile_stats(zvg),
                "z_bottom_m": self._percentile_stats(zvg_min) if zvg_min else None,
            }

        yaws = [o.orientation_yaw for o in obs_list if o.orientation_yaw is not None]
        if len(yaws) >= 10:
            entry["orientation"] = {
                "yaw_mean_deg": round(float(np.mean(yaws)), 1),
                "yaw_std_deg": round(float(np.std(yaws)), 1),
                "n_samples": int(len(yaws)),
            }

        return entry

    def _aggregate_orientation_relationships(self) -> Dict[str, dict]:
        BACK_TO_WALL_THRESHOLD = 45.0
        FACES_THRESHOLD = 60.0
        MIN_SAMPLES = 5

        by_cat: Dict[str, List[OrientationObservation]] = defaultdict(list)
        for oo in self.orientation_observations:
            by_cat[oo.category].append(oo)

        result: Dict[str, dict] = {}
        for cat, oo_list in by_cat.items():
            entry: dict = {}

            wall_angles = [oo.angle_to_nearest_wall for oo in oo_list if oo.angle_to_nearest_wall is not None]
            if len(wall_angles) >= MIN_SAMPLES:
                n_back = sum(1 for a in wall_angles if a < BACK_TO_WALL_THRESHOLD)
                entry["back_to_wall"] = {
                    "fraction": round(n_back / len(wall_angles), 3),
                    "mean_angle": round(float(np.mean(wall_angles)), 1),
                    "n_samples": int(len(wall_angles)),
                }

            center_angles = [oo.angle_to_room_center for oo in oo_list if oo.angle_to_room_center is not None]
            if len(center_angles) >= MIN_SAMPLES:
                n_faces = sum(1 for a in center_angles if a < FACES_THRESHOLD)
                entry["faces_center"] = {
                    "fraction": round(n_faces / len(center_angles), 3),
                    "mean_angle": round(float(np.mean(center_angles)), 1),
                    "n_samples": int(len(center_angles)),
                }

            pair_data: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
            for oo in oo_list:
                for other_cat, angle, dist in oo.facing_pairs:
                    pair_data[other_cat].append((angle, dist))

            faces_pair: Dict[str, dict] = {}
            for other_cat, angle_dist_list in pair_data.items():
                if len(angle_dist_list) < MIN_SAMPLES:
                    continue
                angles = [a for a, _ in angle_dist_list]
                dists = [d for _, d in angle_dist_list]
                n_facing = sum(1 for a in angles if a < FACES_THRESHOLD)
                faces_pair[other_cat] = {
                    "fraction": round(n_facing / len(angles), 3),
                    "mean_angle": round(float(np.mean(angles)), 1),
                    "mean_dist": round(float(np.mean(dists)), 2),
                    "n_samples": int(len(angles)),
                }

            if faces_pair:
                sorted_pairs = sorted(faces_pair.items(), key=lambda x: -x[1]["fraction"])
                entry["faces_pair"] = dict(sorted_pairs[:10])

            if entry:
                result[cat] = entry

        log.info(f"Orientation relationships computed for {len(result)} categories")
        return result

    def _percentile_stats(self, values: List[float]) -> dict:
        arr = np.array(values, dtype=np.float64)
        return {
            "p5": round(float(np.percentile(arr, 5)), 3),
            "p25": round(float(np.percentile(arr, 25)), 3),
            "median": round(float(np.median(arr)), 3),
            "p75": round(float(np.percentile(arr, 75)), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "mean": round(float(np.mean(arr)), 3),
            "std": round(float(np.std(arr)), 3),
        }

    def _compute_cooccurrence_normalized(self):
        count_a_global = Counter()
        count_ab_global: Dict[str, Counter] = defaultdict(Counter)

        count_a_room: Dict[str, Counter] = defaultdict(Counter)
        count_ab_room: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

        scenes_by_room: Counter = Counter()

        for scene in self.scenes:
            cats = set(scene.object_categories)
            if not cats:
                continue
            room = scene.room_type or "unknown"
            scenes_by_room[room] += 1

            for a in cats:
                count_a_global[a] += 1
                count_a_room[room][a] += 1
            for a in cats:
                for b in cats:
                    if a == b:
                        continue
                    count_ab_global[a][b] += 1
                    count_ab_room[room][a][b] += 1

        def make_stats(count_a: Counter, count_ab: Dict[str, Counter], n_scenes: int):
            out: Dict[str, Dict[str, dict]] = defaultdict(dict)
            if n_scenes <= 0:
                return out
            p_a = {a: count_a[a] / n_scenes for a in count_a}

            for a, b_counts in count_ab.items():
                for b, cab in b_counts.items():
                    if cab <= 0:
                        continue
                    p_ab = cab / n_scenes
                    pa = p_a.get(a, 0.0)
                    pb = p_a.get(b, 0.0)
                    p_b_given_a = cab / count_a[a] if count_a[a] > 0 else 0.0

                    npmi = None
                    if pa > 0 and pb > 0 and p_ab > 0:
                        pmi = math.log(p_ab / (pa * pb))
                        denom = -math.log(p_ab)
                        if denom > 1e-12:
                            npmi = pmi / denom
                            npmi = max(-1.0, min(1.0, float(npmi)))
                        else:
                            npmi = 1.0

                    out[a][b] = {
                        "count": int(cab),
                        "p_b_given_a": round(float(p_b_given_a), 4),
                        "npmi": round(float(npmi), 4) if npmi is not None else None,
                    }
            return out

        global_stats = make_stats(count_a_global, count_ab_global, len(self.scenes))

        per_room_stats: Dict[str, Dict[str, Dict[str, dict]]] = defaultdict(dict)
        for room, ca in count_a_room.items():
            n_room = int(scenes_by_room[room])
            room_stats = make_stats(ca, count_ab_room[room], n_room)
            for a, stats in room_stats.items():
                per_room_stats[a][room] = stats

        return global_stats, per_room_stats

def export_ontology(ontology: Dict[str, dict], output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "object_ontology.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ontology, f, indent=2)
    log.info(f"Wrote full ontology -> {json_path}")

    csv_path = out / "object_ontology_summary.csv"
    rows: List[dict] = []

    for cat, entry in sorted(ontology.items()):
        row: dict = {"category": cat, "count": entry.get("count", 0)}
        row["sources"] = ",".join(entry.get("sources", []))

        dims = entry.get("dimensions", {})
        if dims:
            for axis in ["width_m", "height_m", "depth_m"]:
                if dims.get(axis):
                    row[f"{axis}_median"] = dims[axis]["median"]
                    row[f"{axis}_p5"] = dims[axis]["p5"]
                    row[f"{axis}_p95"] = dims[axis]["p95"]
            row["dim_samples_w"] = dims.get("n_width", 0)
            row["dim_samples_h"] = dims.get("n_height", 0)
            row["dim_samples_d"] = dims.get("n_depth", 0)

        rooms = entry.get("room_associations", {})
        row["top_rooms"] = "; ".join(
            f"{r}({v['fraction']:.0%})"
            for r, v in sorted(rooms.items(), key=lambda x: -x[1]["fraction"])[:3]
        )

        supports = entry.get("support_surfaces", {})
        row["top_supports"] = "; ".join(
            f"{s}({v['fraction']:.0%})"
            for s, v in sorted(supports.items(), key=lambda x: -x[1]["fraction"])[:3]
        )

        cooccur = entry.get("cooccurrence", {}) or {}
        row["top_cooccurrences"] = "; ".join(
            f"{k}(c={v.get('count')},p={v.get('p_b_given_a')},npmi={v.get('npmi')})"
            for k, v in sorted(cooccur.items(), key=lambda x: -(x[1].get("count", 0)))[:5]
        )

        hb3 = entry.get("height_band_3d", {})
        if hb3 and hb3.get("z_center_m"):
            row["height3d_median"] = hb3["z_center_m"]["median"]
        hbv = entry.get("height_band_vg", {})
        if hbv and hbv.get("z_center_m"):
            row["heightvg_median"] = hbv["z_center_m"]["median"]

        orient_rel = entry.get("orientation_relationships", {}) or {}
        btw = orient_rel.get("back_to_wall", {})
        if btw:
            row["back_to_wall_frac"] = btw.get("fraction", "")
        fc = orient_rel.get("faces_center", {})
        if fc:
            row["faces_center_frac"] = fc.get("fraction", "")
        fp = orient_rel.get("faces_pair", {})
        if fp:
            row["top_faces_pair"] = "; ".join(
                f"{k}({v['fraction']:.0%})"
                for k, v in sorted(fp.items(), key=lambda x: -x[1]["fraction"])[:3]
            )

        rows.append(row)

    if rows:
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        fieldnames = sorted(all_keys)
        if "category" in fieldnames:
            fieldnames.insert(0, fieldnames.pop(fieldnames.index("category")))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"Wrote summary CSV -> {csv_path}")

    domain_configs = generate_verifier_configs(ontology)
    for domain, config in domain_configs.items():
        cfg_path = out / f"verifier_config_{domain}.json"
        with open(cfg_path, "w") as f:
            json.dump(config, f, indent=2)
        log.info(f"Wrote verifier config -> {cfg_path}")

    report_path = out / "mining_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("Object Statistics Mining Report\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total categories:     {len(ontology)}\n")
        total_obs = sum(int(e.get("count", 0)) for e in ontology.values())
        f.write(f"Total observations:   {total_obs}\n")
        cats_with_dims = sum(1 for e in ontology.values() if "dimensions" in e)
        f.write(f"Categories with dims: {cats_with_dims}\n")
        cats_with_orient = sum(1 for e in ontology.values() if "orientation_relationships" in e)
        f.write(f"Categories with orient rel: {cats_with_orient}\n\n")

        f.write("Top 20 categories by observation count:\n")
        top20 = sorted(ontology.items(), key=lambda x: -int(x[1].get("count", 0)))[:20]
        for cat, entry in top20:
            f.write(f"  {cat:25s}  {int(entry.get('count', 0)):>6d}  sources={entry.get('sources', [])}\n")

        f.write("\nOrientation relationship highlights:\n")
        for cat, entry in sorted(ontology.items()):
            orient_rel = entry.get("orientation_relationships", {}) or {}
            if not orient_rel:
                continue
            btw = orient_rel.get("back_to_wall", {}) or {}
            fc = orient_rel.get("faces_center", {}) or {}
            fp = orient_rel.get("faces_pair", {}) or {}

            highlights = []
            if btw and btw.get("fraction", 0) > 0.5:
                highlights.append(f"back_to_wall={btw['fraction']:.0%}")
            if fc and fc.get("fraction", 0) > 0.5:
                highlights.append(f"faces_center={fc['fraction']:.0%}")
            if fp:
                top_fp = sorted(fp.items(), key=lambda x: -x[1]["fraction"])[:2]
                for k, v in top_fp:
                    if v["fraction"] > 0.5:
                        highlights.append(f"faces_{k}={v['fraction']:.0%}")
            if highlights:
                f.write(f"  {cat:25s}  {', '.join(highlights)}\n")

    log.info(f"Wrote report -> {report_path}")

def generate_verifier_configs(ontology):
    domain_object_pools = {
        "bedroom": {
            "bed", "nightstand", "wardrobe", "dresser", "lamp", "table_lamp",
            "floor_lamp", "rug", "mirror", "chair", "armchair", "desk",
            "bookshelf", "curtain", "painting", "plant", "pillow", "blanket",
            "clock", "photo_frame", "vase", "ceiling_lamp", "wall_lamp",
            "ceiling_fan", "shelf", "wall_shelf", "bench", "ottoman",
            "tv", "tv_stand", "stool",
        },
        "living_room": {
            "sofa", "coffee_table", "armchair", "tv", "tv_stand", "rug",
            "floor_lamp", "table_lamp", "bookshelf", "side_table", "plant",
            "painting", "curtain", "console_table", "ottoman", "lamp",
            "chair", "ceiling_lamp", "chandelier", "pendant_lamp",
            "wall_lamp", "shelf", "wall_shelf", "clock", "vase",
            "photo_frame", "speaker", "pillow", "blanket", "fireplace",
            "sideboard", "cabinet", "recliner", "bench", "stool",
            "ceiling_fan",
        },
        "bookstore": {
            "bookshelf", "shelf", "wall_shelf", "table", "desk", "chair",
            "armchair", "bench", "lamp", "ceiling_lamp", "pendant_lamp",
            "rug", "painting", "clock", "plant",
        },
        "buffet_restaurant": {
            "dining_table", "chair", "bench", "sideboard", "cabinet",
            "counter", "lamp", "ceiling_lamp", "pendant_lamp",
            "painting", "plant", "rug",
        },
        "children_room": {
            "bed", "crib", "bunk_bed", "nightstand", "dresser", "wardrobe",
            "bookshelf", "shelf", "chair", "rug", "lamp", "table_lamp",
            "curtain", "painting", "plant", "pillow", "blanket",
        },
        "classroom": {
            "desk", "chair", "table", "bookshelf", "shelf",
            "monitor", "computer", "printer",
            "clock", "painting",
        },
        "computer_room": {
            "desk", "office_chair", "chair",
            "monitor", "computer", "keyboard", "printer", "speaker",
            "shelf", "cabinet", "filing_cabinet", "lamp",
        },
        "deli": {
            "counter", "cabinet", "shelf", "refrigerator",
            "table", "chair", "lamp", "ceiling_lamp",
        },
        "dining_room": {
            "dining_table", "chair", "sideboard", "cabinet",
            "pendant_lamp", "ceiling_lamp", "rug", "painting", "plant",
        },
        "florist_shop": {
            "plant", "planter", "vase",
            "shelf", "wall_shelf", "table", "counter", "cabinet",
            "chair", "lamp", "ceiling_lamp", "painting",
        },
        "game_room": {
            "sofa", "armchair", "chair", "coffee_table", "side_table",
            "tv", "tv_stand", "speaker", "rug", "lamp", "ceiling_lamp",
            "bookshelf", "shelf",
        },
    }

    def pick_height_band(entry: dict):
        hb3 = entry.get("height_band_3d")
        if isinstance(hb3, dict) and hb3.get("z_center_m"):
            return hb3
        hbv = entry.get("height_band_vg")
        if isinstance(hbv, dict) and hbv.get("z_center_m"):
            return hbv
        return None

    configs = {}
    for domain, valid_cats in domain_object_pools.items():
        config = {
            "domain": domain,
            "valid_objects": {},
            "support_rules": {},
            "dimension_ranges": {},
            "height_bands": {},
        }

        for cat in valid_cats:
            if cat not in ontology:
                continue
            entry = ontology[cat]

            dims = entry.get("dimensions", {}) or {}
            if dims:
                dim_range = {}
                for axis in ["width_m", "height_m", "depth_m"]:
                    if dims.get(axis):
                        dim_range[axis.replace("_m", "")] = {
                            "min": dims[axis]["p5"],
                            "max": dims[axis]["p95"],
                        }
                if dim_range:
                    config["dimension_ranges"][cat] = dim_range

            supports = entry.get("support_surfaces", {}) or {}
            if supports:
                config["support_rules"][cat] = [
                    s for s, v in supports.items()
                    if float(v.get("fraction", 0.0)) >= 0.05
                ]

            hb = pick_height_band(entry)
            if hb and hb.get("z_bottom_m") and hb.get("z_center_m"):
                config["height_bands"][cat] = {
                    "z_min": hb["z_bottom_m"]["p5"],
                    "z_max": hb["z_center_m"]["p95"],
                }

            config["valid_objects"][cat] = True

        configs[domain] = config

    return configs

def main():
    parser = argparse.ArgumentParser(description="Mine object statistics from 3D-FRONT, ScanNet, and Visual Genome")
    parser.add_argument("--threedfront_dir", type=str, default=None)
    parser.add_argument("--model_info", type=str, default=None)
    parser.add_argument("--scannet_dir", type=str, default=None)
    parser.add_argument("--vg_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./ontology_output")
    args = parser.parse_args()

    if not any([args.threedfront_dir, args.scannet_dir, args.vg_dir]):
        log.error("At least one dataset path must be provided.")
        parser.print_help()
        sys.exit(1)

    builder = OntologyBuilder()

    if args.threedfront_dir:
        log.info(f"=== Parsing 3D-FRONT from {args.threedfront_dir} ===")
        tdp = ThreeDFrontParser(args.threedfront_dir, model_info_path=args.model_info)
        obs, scenes, orient_obs = tdp.parse_all()
        builder.add(obs, scenes, orient_obs)

    if args.scannet_dir:
        log.info(f"=== Parsing ScanNet from {args.scannet_dir} ===")
        snp = ScanNetParser(args.scannet_dir)
        obs, scenes = snp.parse_all()
        builder.add(obs, scenes)

    if args.vg_dir:
        log.info(f"=== Parsing Visual Genome from {args.vg_dir} ===")
        vgp = VisualGenomeParser(args.vg_dir)
        obs, scenes = vgp.parse_all()
        builder.add(obs, scenes)

    ontology = builder.build()
    export_ontology(ontology, args.output_dir)
    log.info("Done!")

if __name__ == "__main__":
    main()