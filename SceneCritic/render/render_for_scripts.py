import bpy
import os
import math
import copy
from pathlib import Path
from mathutils import Vector
import json
import re
from difflib import SequenceMatcher

MESH_DIR = "/path/to/mesh_directory"

DEFAULT_HEIGHTS = {
    'bed': 0.0,
    'double_bed': 0.0,
    'cabinet': 0.0,
    'desk': 0.0,
    'chair': 0.0,
    'shelf': 0.0,
}

CAMERA_CONFIG = {
    'top_height_factor': 1.3,
    'top_focal_length': 27,
    'top_sensor_size': 30,
    'orbit_height': 2.5,
    'orbit_radius_factor': 1.2,
    'orbit_tilt': math.radians(75),
    'orbit_focal_length': 25,
}

FLOOR_COLOR_HEX = "E7C388"

MESH_EXTENSIONS = {'.glb', '.gltf', '.obj', '.fbx', '.blend'}


def scan_mesh_directory(mesh_dir):
    available_meshes = {}
    
    if not os.path.isdir(mesh_dir):
        print(f"Warning: Mesh directory not found: {mesh_dir}")
        return available_meshes
    
    for filename in os.listdir(mesh_dir):
        filepath = os.path.join(mesh_dir, filename)
        if os.path.isfile(filepath):
            ext = Path(filename).suffix.lower()
            if ext in MESH_EXTENSIONS:
                base_name = Path(filename).stem.lower()
                available_meshes[base_name] = filename
                
    print(f"Found {len(available_meshes)} mesh files in directory: {list(available_meshes.keys())}")
    return available_meshes


def normalize_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())


def get_name_components(name):
    name_lower = name.lower()
    parts = re.split(r'[_\-\s]+', name_lower)
    expanded = []
    for part in parts:
        camel_split = re.findall(r'[a-z]+|[0-9]+', part)
        expanded.extend(camel_split)
    
    common_suffixes = ['bed', 'chair', 'table', 'desk', 'shelf', 'case', 'stand', 
                       'board', 'cabinet', 'sofa', 'couch', 'lamp', 'light', 'rug',
                       'mat', 'mirror', 'drawer', 'wardrobe', 'closet', 'bench']
    common_prefixes = ['book', 'night', 'bed', 'side', 'arm', 'dining', 'coffee',
                       'end', 'tv', 'office', 'desk', 'floor', 'wall', 'double',
                       'single', 'king', 'queen', 'twin', 'corner', 'round', 'bar']
    
    additional = []
    for part in expanded:
        if len(part) > 4:
            for suffix in common_suffixes:
                if part.endswith(suffix) and len(part) > len(suffix):
                    prefix = part[:-len(suffix)]
                    if len(prefix) > 2:
                        additional.extend([prefix, suffix])
            for prefix in common_prefixes:
                if part.startswith(prefix) and len(part) > len(prefix):
                    suffix = part[len(prefix):]
                    if len(suffix) > 2:
                        additional.extend([prefix, suffix])
    
    expanded.extend(additional)
    return list(set(p for p in expanded if p and len(p) > 1))


def compute_match_score(query_name, candidate_name):
    q_norm = normalize_name(query_name)
    c_norm = normalize_name(candidate_name)
    
    if q_norm == c_norm:
        return 100
    
    if q_norm in c_norm:
        coverage = len(q_norm) / len(c_norm)
        return 80 + (coverage * 15)
    
    if c_norm in q_norm:
        coverage = len(c_norm) / len(q_norm)
        return 75 + (coverage * 15)
    
    q_components = set(get_name_components(query_name))
    c_components = set(get_name_components(candidate_name))
    
    if q_components and c_components:
        shared = q_components & c_components
        if shared:
            match_ratio = len(shared) / max(len(q_components), len(c_components))
            return 60 + (match_ratio * 25)
    
    similarity = SequenceMatcher(None, q_norm, c_norm).ratio()
    return similarity * 50


def find_best_mesh_match(obj_type, available_meshes, mesh_files=None, threshold=40):
    if mesh_files and obj_type in mesh_files:
        return mesh_files[obj_type], 100
    
    obj_type_lower = obj_type.lower()
    if obj_type_lower in available_meshes:
        return available_meshes[obj_type_lower], 100
    
    best_match = None
    best_score = 0
    
    for mesh_name, mesh_file in available_meshes.items():
        score = compute_match_score(obj_type, mesh_name)
        if score > best_score:
            best_score = score
            best_match = mesh_file
    
    if best_score >= threshold:
        print(f"  Fuzzy match: '{obj_type}' -> '{best_match}' (score: {best_score:.1f})")
        return best_match, best_score
    
    return None, 0


def build_mesh_mapping(obj_types, mesh_dir, mesh_files=None, threshold=40):
    available_meshes = scan_mesh_directory(mesh_dir)
    mapping = {}
    unmatched = []
    
    for obj_type in obj_types:
        mesh_file, score = find_best_mesh_match(
            obj_type, available_meshes, mesh_files, threshold
        )
        if mesh_file:
            mapping[obj_type] = mesh_file
        else:
            unmatched.append(obj_type)
    
    if unmatched:
        print(f"Warning: No mesh files found for: {unmatched}")
    
    return mapping


def normalize_room_size(room_size):
    if isinstance(room_size, (int, float)):
        return (float(room_size), float(room_size))
    elif isinstance(room_size, (tuple, list)) and len(room_size) >= 2:
        return (float(room_size[0]), float(room_size[1]))
    else:
        raise ValueError(f"Invalid room_size format: {room_size}.")


def extract_mesh_dimensions(mesh_dir, mesh_files):
    sizes = {}
    
    for obj_type, mesh_file in mesh_files.items():
        filepath = os.path.join(mesh_dir, mesh_file)
        
        if not os.path.exists(filepath):
            print(f"Mesh file not found: {filepath}, using default size [1.0, 1.0, 1.0]")
            sizes[obj_type] = [1.0, 1.0, 1.0]
            continue
        
        imported = import_mesh(filepath)
        
        if not imported:
            sizes[obj_type] = [1.0, 1.0, 1.0]
            continue
        
        min_coords = [float('inf')] * 3
        max_coords = [float('-inf')] * 3
        
        for obj in imported:
            if obj.type == 'MESH':
                bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                for corner in bbox_corners:
                    for i in range(3):
                        min_coords[i] = min(min_coords[i], corner[i])
                        max_coords[i] = max(max_coords[i], corner[i])
        
        if min_coords[0] != float('inf'):
            dims = [max_coords[i] - min_coords[i] for i in range(3)]
            sizes[obj_type] = dims
            print(f"Extracted dimensions for '{obj_type}': {dims}")
        else:
            sizes[obj_type] = [1.0, 1.0, 1.0]
            print(f"Could not extract dimensions for '{obj_type}', using default")
        
        for obj in imported:
            bpy.data.objects.remove(obj, do_unlink=True)
    
    return sizes


def extract_mesh_dimensions_auto(mesh_dir, obj_types, mesh_files=None, threshold=40):
    mapping = build_mesh_mapping(obj_types, mesh_dir, mesh_files, threshold)
    return extract_mesh_dimensions(mesh_dir, mapping), mapping


def parse_format1(furniture_data, object_sizes=None, mesh_dir=None, mesh_files=None):
    if object_sizes is None:
        if mesh_dir is None or mesh_files is None:
            raise ValueError("Either object_sizes or (mesh_dir and mesh_files) must be provided")
        print("Extracting object sizes from mesh files...")
        object_sizes = extract_mesh_dimensions(mesh_dir, mesh_files)
    
    objects = []
    
    for name, data in furniture_data.items():
        obj_type = name.rsplit('-', 1)[0]
        
        x, y = data['position'][0], data['position'][1]
        z = 0
        
        rotation_deg = data['rotation'][2]
        
        width, depth, height = object_sizes.get(obj_type, [1.0, 1.0, 1.0])
        
        objects.append({
            'name': name,
            'type': obj_type,
            'position': (x, y, z),
            'rotation_deg': rotation_deg,
            'dimensions': (width, depth),
        })
    
    return objects


def parse_format2(furniture_data, room_size=None):
    objects = []
    object_list = furniture_data[0]['object_list']
    
    obj_counts = {}
    for obj_name, obj_data in object_list:
        if obj_name not in obj_counts:
            obj_counts[obj_name] = 0
        instance_name = f"{obj_name}-{obj_counts[obj_name]}"
        obj_counts[obj_name] += 1
        
        length = obj_data['length']
        width = obj_data['width']
        height = obj_data.get('height', 1.0)
        left = obj_data['left']
        depth = obj_data['depth']
        orientation = obj_data['orientation']
        
        orientation = orientation % 360
        
        if orientation in [0, 180]:
            bbox_x = length
            bbox_y = width
        else:
            bbox_x = width
            bbox_y = length
        
        cx = left + bbox_x / 2
        cy = depth + bbox_y / 2
        
        objects.append({
            'name': instance_name,
            'type': obj_name,
            'position': (cx, cy, 0),
            'rotation_deg': orientation,
            'dimensions': (length, width),
            'height': height,
        })
    
    return objects


def infer_room_size_format2(furniture_data, padding=0.5):
    object_list = furniture_data[0]['object_list']
    
    if not object_list:
        return (5.0, 5.0)
    
    max_x = 0
    max_y = 0
    
    for obj_name, obj_data in object_list:
        length = obj_data['length']
        width = obj_data['width']
        left = obj_data['left']
        depth = obj_data['depth']
        orientation = obj_data['orientation']
        
        if orientation == 0 or orientation == 180:
            right = left + length
            top = depth + width
        else:
            right = left + width
            top = depth + length
        
        max_x = max(max_x, right)
        max_y = max(max_y, top)
    
    return (max_x + padding, max_y + padding)


def normalize_format3_to_room(furniture_data, room_size=(5.0, 5.0), margin=0.1):
    room_width, room_depth = normalize_room_size(room_size)
    normalized_data = copy.deepcopy(furniture_data)
    objects = normalized_data['objects']
    
    valid_objects = [obj for obj in objects 
                     if 'ego' not in obj['label'] and 
                     not (obj['bbox'][2] == 0 and obj['bbox'][3] == 0)]
    
    if not valid_objects:
        return normalized_data
    
    min_x = min(obj['bbox'][0] for obj in valid_objects)
    max_x = max(obj['bbox'][0] + obj['bbox'][2] for obj in valid_objects)
    min_y = min(obj['bbox'][1] for obj in valid_objects)
    max_y = max(obj['bbox'][1] + obj['bbox'][3] for obj in valid_objects)
    
    span_x = max_x - min_x
    span_y = max_y - min_y
    available_width = room_width - 2 * margin
    available_depth = room_depth - 2 * margin
    
    if span_x > 0 and span_y > 0:
        scale = min(available_width / span_x, available_depth / span_y)
        if max(span_x, span_y) <= min(available_width, available_depth):
            scale = 1.0
    else:
        scale = 1.0
    
    offset_x = -min_x * scale + margin
    offset_y = -min_y * scale + margin
    
    if span_x * scale < available_width:
        offset_x += (available_width - span_x * scale) / 2
    if span_y * scale < available_depth:
        offset_y += (available_depth - span_y * scale) / 2
    
    for obj in normalized_data['objects']:
        old_pos = obj['position']
        obj['position'] = [old_pos[0] * scale + offset_x, old_pos[1] * scale + offset_y]
        
        old_bbox = obj['bbox']
        obj['bbox'] = [
            old_bbox[0] * scale + offset_x,
            old_bbox[1] * scale + offset_y,
            old_bbox[2] * scale,
            old_bbox[3] * scale
        ]
    
    normalized_data['_normalization'] = {
        'scale': scale, 
        'offset': [offset_x, offset_y],
        'room_size': [room_width, room_depth]
    }
    return normalized_data


def parse_format3(furniture_data, normalize=True, room_size=(5.0, 5.0)):
    room_size = normalize_room_size(room_size)
    
    if normalize:
        furniture_data = normalize_format3_to_room(furniture_data, room_size=room_size)
    
    objects = []
    
    for obj in furniture_data['objects']:
        label = obj['label']
        bbox = obj['bbox']
        position = obj['position']
        rotation_angle = obj.get('rotation_angle', 0.0)
        
        obj_type = label.rsplit('_', 1)[0]
        
        if 'ego' in obj_type or (bbox[2] == 0 and bbox[3] == 0):
            continue
        
        cx, cy = position
        width, height = bbox[2], bbox[3]
        z = DEFAULT_HEIGHTS.get(obj_type, 0)
        
        objects.append({
            'name': label,
            'type': obj_type,
            'position': (cx, cy, z),
            'rotation_deg': rotation_angle,
            'dimensions': (width, height),
        })
    
    return objects


def parse_format3_v2(furniture_data, normalize=True, room_size=(5.0, 5.0), 
                     object_sizes=None, mesh_dir=None, mesh_files=None):
    if object_sizes is None:
        if mesh_dir is None or mesh_files is None:
            raise ValueError("Either object_sizes or (mesh_dir and mesh_files) must be provided")
        print("Extracting object sizes from mesh files...")
        object_sizes = extract_mesh_dimensions(mesh_dir, mesh_files)
    
    room_size = normalize_room_size(room_size)
    
    if normalize:
        furniture_data = normalize_format3_to_room(furniture_data, room_size=room_size)
    
    objects = []
    
    for obj in furniture_data['objects']:
        label = obj['label']
        bbox = obj['bbox']
        position = obj['position']
        rotation_angle = obj.get('rotation_angle', 0.0)
        
        obj_type = label.rsplit('_', 1)[0]
        
        if 'ego' in obj_type or (bbox[2] == 0 and bbox[3] == 0):
            continue
        
        cx, cy = position
        z = DEFAULT_HEIGHTS.get(obj_type, 0)
        
        mesh_dims = object_sizes.get(obj_type, [1.0, 1.0, 1.0])
        width, depth = mesh_dims[0], mesh_dims[1]
        
        objects.append({
            'name': label,
            'type': obj_type,
            'position': (cx, cy, z),
            'rotation_deg': rotation_angle,
            'dimensions': (width, depth),
        })
    
    return objects


def infer_room_size_format3(furniture_data, padding=0.5):
    objects = furniture_data.get('objects', [])
    
    valid_objects = [obj for obj in objects 
                     if 'ego' not in obj['label'] and 
                     not (obj['bbox'][2] == 0 and obj['bbox'][3] == 0)]
    
    if not valid_objects:
        return (5.0, 5.0)
    
    max_x = max(obj['bbox'][0] + obj['bbox'][2] for obj in valid_objects)
    max_y = max(obj['bbox'][1] + obj['bbox'][3] for obj in valid_objects)
    
    return (max_x + padding, max_y + padding)


def infer_room_size_format1(furniture_data, object_sizes, padding=0.5):
    if not furniture_data:
        return (5.0, 5.0)
    
    max_x = 0
    max_y = 0
    
    for name, data in furniture_data.items():
        obj_type = name.rsplit('-', 1)[0]
        x, y = data['position'][0], data['position'][1]
        
        dims = object_sizes.get(obj_type, [1.0, 1.0, 1.0])
        width, depth = dims[0], dims[1]
        
        max_dim = max(width, depth)
        max_x = max(max_x, x + max_dim / 2)
        max_y = max(max_y, y + max_dim / 2)
    
    return (max_x + padding, max_y + padding)


def _holodeck_extract_room_bounds(furniture_data):
    rooms = furniture_data.get('rooms', [])
    if not rooms:
        return None
    
    room = rooms[0]
    vertices = room.get('vertices', room.get('full_vertices', []))
    
    if not vertices:
        return None
    
    xs = [v[0] for v in vertices]
    zs = [v[1] for v in vertices]
    
    return (min(xs), min(zs), max(xs), max(zs))


def _holodeck_vertices_to_dimensions_meters(vertices):
    if not vertices or len(vertices) < 2:
        return (1.0, 1.0)
    
    xs = [v[0] for v in vertices]
    zs = [v[1] for v in vertices]
    
    width_cm = max(xs) - min(xs)
    depth_cm = max(zs) - min(zs)
    
    width_m = width_cm / 100.0
    depth_m = depth_cm / 100.0
    
    return (width_m, depth_m)


def parse_format4(furniture_data, room_size=None):
    obj_list = furniture_data.get('floor_objects', furniture_data.get('objects', []))
    
    bounds = _holodeck_extract_room_bounds(furniture_data)
    
    if bounds is not None:
        min_x, min_z, max_x, max_z = bounds
        offset_x = -min_x
        offset_z = -min_z
        room_width = max_x - min_x
        room_depth = max_z - min_z
    else:
        offset_x = 2.5
        offset_z = 2.5
        room_width = 5.0
        room_depth = 5.0
    
    if room_size is not None:
        rw, rd = normalize_room_size(room_size)
        scale_x = rw / room_width if room_width > 0 else 1.0
        scale_z = rd / room_depth if room_depth > 0 else 1.0
    else:
        scale_x = 1.0
        scale_z = 1.0
    
    objects = []
    
    for obj in obj_list:
        object_name = obj.get('object_name', '')
        obj_id = obj.get('id', '')
        
        if not object_name:
            object_name = obj_id.split(' ')[0] if obj_id else ''
        
        if not object_name:
            continue
        
        obj_type = object_name.rsplit('-', 1)[0]
        
        pos = obj.get('position', {})
        holo_x = pos.get('x', 0.0)
        holo_y = pos.get('y', 0.0)
        holo_z = pos.get('z', 0.0)
        
        blender_x = (holo_x + offset_x) * scale_x
        blender_y = (holo_z + offset_z) * scale_z
        blender_z = 0.0
        
        rot = obj.get('rotation', {})
        holo_rot_y = rot.get('y', 0.0)
        
        blender_rot_deg = -holo_rot_y
        
        objects.append({
            'name': object_name,
            'type': obj_type,
            'position': (blender_x, blender_y, blender_z),
            'rotation_deg': blender_rot_deg,
            'dimensions': (0, 0),
            'native_scale': True,
        })
        
        print(f"  Holodeck parsed: {object_name} "
              f"holo=({holo_x:.2f}, {holo_y:.2f}, {holo_z:.2f}) "
              f"-> blender=({blender_x:.2f}, {blender_y:.2f}, {blender_z:.2f}) "
              f"rot={blender_rot_deg:.1f}° (native scale, no resize)")
    
    return objects


def infer_room_size_format4(furniture_data):
    bounds = _holodeck_extract_room_bounds(furniture_data)
    
    if bounds is not None:
        min_x, min_z, max_x, max_z = bounds
        return (max_x - min_x, max_z - min_z)
    
    obj_list = furniture_data.get('floor_objects', furniture_data.get('objects', []))
    
    if not obj_list:
        return (5.0, 5.0)
    
    max_x = 0
    max_z = 0
    
    for obj in obj_list:
        pos = obj.get('position', {})
        vertices = obj.get('vertices', [])
        
        x = abs(pos.get('x', 0))
        z = abs(pos.get('z', 0))
        
        if vertices:
            w, d = _holodeck_vertices_to_dimensions_meters(vertices)
            x += w / 2
            z += d / 2
        
        max_x = max(max_x, x)
        max_z = max(max_z, z)
    
    return (max_x * 2 + 0.5, max_z * 2 + 0.5)


def detect_holodeck_format(data):
    if not isinstance(data, dict):
        return False
    
    holodeck_keys = {'rooms', 'walls', 'doors', 'windows', 'metadata'}
    matching_keys = holodeck_keys & set(data.keys())
    
    if len(matching_keys) >= 3:
        return True
    
    objects = data.get('objects', data.get('floor_objects', []))
    if objects and isinstance(objects, list) and len(objects) > 0:
        first_obj = objects[0]
        pos = first_obj.get('position', None)
        if isinstance(pos, dict) and 'x' in pos and 'y' in pos and 'z' in pos:
            if 'assetId' in first_obj and 'roomId' in first_obj:
                return True
    
    return False


def clear_scene(keep_camera=False, keep_lights=False):
    bpy.ops.object.select_all(action='DESELECT')
    
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            obj.select_set(True)
        elif obj.type == 'CAMERA' and not keep_camera:
            obj.select_set(True)
        elif obj.type == 'LIGHT' and not keep_lights:
            obj.select_set(True)
        elif obj.type == 'EMPTY':
            obj.select_set(True)
    
    bpy.ops.object.delete()


def import_mesh(filepath):
    ext = Path(filepath).suffix.lower()
    existing_objects = set(bpy.data.objects)
    
    if ext == '.glb' or ext == '.gltf':
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == '.blend':
        with bpy.data.libraries.load(filepath) as (data_from, data_to):
            data_to.objects = data_from.objects
        for obj in data_to.objects:
            if obj is not None:
                bpy.context.collection.objects.link(obj)
    else:
        print(f"Unsupported file format: {ext}")
        return None
    
    new_objects = set(bpy.data.objects) - existing_objects
    return list(new_objects)


def get_mesh_dimensions(obj):
    if obj.type != 'MESH':
        return (1, 1, 1)
    return tuple(obj.dimensions)


def place_furniture_object(obj_data, mesh_dir, mesh_files, available_meshes=None, 
                           scale_to_fit=True, match_threshold=40):
    obj_type = obj_data['type']
    
    mesh_file = mesh_files.get(obj_type) if mesh_files else None
    
    if not mesh_file and available_meshes:
        mesh_file, score = find_best_mesh_match(
            obj_type, available_meshes, mesh_files, match_threshold
        )
    
    if not mesh_file:
        print(f"No mesh file found for type: {obj_type}")
        return None
    
    filepath = os.path.join(mesh_dir, mesh_file)
    
    if not os.path.exists(filepath):
        print(f"Mesh file not found: {filepath}")
        return None
    
    imported = import_mesh(filepath)
    if not imported:
        return None
    
    imported_obj = imported[0]
    imported_obj.name = obj_data['name']
    
    use_native_scale = obj_data.get('native_scale', False)
    
    if scale_to_fit and not use_native_scale:
        target_width, target_depth = obj_data['dimensions']
        current_dims = get_mesh_dimensions(imported_obj)
        if current_dims[0] > 0 and current_dims[1] > 0:
            scale_x = target_width / current_dims[0]
            scale_y = target_depth / current_dims[1]
            scale_z = (scale_x + scale_y) / 2
            imported_obj.scale = (scale_x, scale_y, scale_z)
    elif use_native_scale:
        print(f"  Using native asset scale for {obj_data['name']} (Holodeck mode)")
    
    angle_deg = obj_data.get('rotation_deg', 0.0)
    print("APPLY ROTATION OF: ", angle_deg)
    angle_rad = math.radians(angle_deg)
    bpy.ops.object.select_all(action='DESELECT')
    imported_obj.select_set(True)
    bpy.context.view_layer.objects.active = imported_obj
    imported_obj.rotation_mode = 'XYZ'
    imported_obj.rotation_euler.z = angle_rad
    
    x, y, z = obj_data['position']
    imported_obj.location = (x, y, z)
    
    print(f"Placed {obj_data['name']} at ({x:.2f}, {y:.2f}, {z:.2f}) rot={angle_deg:.1f}°"
          f"{' [native scale]' if use_native_scale else ''}")
    
    return imported_obj


def place_all_furniture(parsed_objects, mesh_dir, mesh_files=None, scale_to_fit=True, 
                        match_threshold=40):
    available_meshes = scan_mesh_directory(mesh_dir)
    
    placed = []
    
    for obj_data in parsed_objects:
        obj = place_furniture_object(
            obj_data, mesh_dir, mesh_files, available_meshes,
            scale_to_fit, match_threshold
        )
        if obj:
            placed.append(obj)
    
    return placed


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b, 1.0)


def create_floor_material(color_hex):
    mat = bpy.data.materials.new(name="FloorMaterial")
    mat.use_nodes = True
    
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs['Base Color'].default_value = hex_to_rgb(color_hex)
        bsdf.inputs['Roughness'].default_value = 0.8
    
    return mat


def create_floor(room_size=(5.0, 5.0), color_hex=None):
    room_width, room_depth = normalize_room_size(room_size)
    
    bpy.ops.mesh.primitive_plane_add(size=1, location=(room_width/2, room_depth/2, 0))
    floor = bpy.context.active_object
    floor.name = "Floor"
    floor.scale = (room_width, room_depth, 1)
    
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    
    if color_hex:
        mat = create_floor_material(color_hex)
        floor.data.materials.append(mat)
    
    return floor


def setup_lighting(room_size=(5.0, 5.0), room_height=2.5):
    room_width, room_depth = normalize_room_size(room_size)
    lights = []
    center_x = room_width / 2
    center_y = room_depth / 2
    
    bpy.ops.object.light_add(type='SUN', location=(center_x, center_y, room_height + 2))
    sun = bpy.context.active_object
    sun.name = "Sun_Main"
    sun.data.energy = 3.0
    sun.data.angle = math.radians(45)
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))
    lights.append(sun)
    
    bpy.ops.object.light_add(type='AREA', location=(center_x, center_y, room_height - 0.1))
    ceiling_light = bpy.context.active_object
    ceiling_light.name = "Ceiling_Area"
    ceiling_light.data.energy = 200.0
    ceiling_light.data.size = min(room_width, room_depth) * 0.8
    ceiling_light.data.shape = 'RECTANGLE'
    ceiling_light.rotation_euler = (math.pi, 0, 0)
    lights.append(ceiling_light)
    
    bpy.ops.object.light_add(type='AREA', location=(room_width + 1, center_y, room_height * 0.6))
    fill_light = bpy.context.active_object
    fill_light.name = "Fill_Side"
    fill_light.data.energy = 100.0
    fill_light.data.size = 2.0
    fill_light.rotation_euler = (math.radians(90), 0, math.radians(-90))
    lights.append(fill_light)
    
    bpy.ops.object.light_add(type='AREA', location=(center_x, -1, room_height * 0.5))
    back_fill = bpy.context.active_object
    back_fill.name = "Fill_Back"
    back_fill.data.energy = 50.0
    back_fill.data.size = 2.0
    back_fill.rotation_euler = (math.radians(70), 0, 0)
    lights.append(back_fill)
    
    return lights


def create_camera(name, location, rotation, lens=35, sensor_size=None):
    bpy.ops.object.camera_add(location=location, rotation=rotation)
    camera = bpy.context.active_object
    camera.name = name
    camera.data.lens = lens
    if sensor_size is not None:
        camera.data.sensor_width = sensor_size
    return camera


def setup_cameras(room_size=(5.0, 5.0), config=None):
    if config is None:
        config = CAMERA_CONFIG
    
    room_width, room_depth = normalize_room_size(room_size)
    cameras = []
    center_x = room_width / 2
    center_y = room_depth / 2
    
    max_dim = max(room_width, room_depth)
    top_height = max_dim * config['top_height_factor']
    orbit_radius = max_dim * config['orbit_radius_factor']
    
    top_camera = create_camera(
        name="Camera_Top",
        location=(center_x, center_y, top_height),
        rotation=(0, 0, 0),
        lens=config['top_focal_length'],
        sensor_size=config['top_sensor_size']
    )
    cameras.append(top_camera)
    
    orbit_angles = [0, 90, 180, 270]
    orbit_height = config['orbit_height']
    orbit_tilt = config['orbit_tilt']
    orbit_focal = config['orbit_focal_length']
    
    for angle_deg in orbit_angles:
        angle_rad = math.radians(angle_deg)
        
        radius_x = room_width / 2 + orbit_radius - max_dim / 2
        radius_y = room_depth / 2 + orbit_radius - max_dim / 2
        
        cam_x = center_x + radius_x * math.cos(angle_rad)
        cam_y = center_y + radius_y * math.sin(angle_rad)
        cam_z = orbit_height
        
        rot_x = orbit_tilt
        rot_z = angle_rad + math.pi / 2
        
        orbit_camera = create_camera(
            name=f"Camera_Orbit_{angle_deg}",
            location=(cam_x, cam_y, cam_z),
            rotation=(rot_x, 0, rot_z),
            lens=orbit_focal
        )
        cameras.append(orbit_camera)
    
    bpy.context.scene.camera = cameras[0]
    
    return cameras


def setup_cycles_gpu(samples=128, use_denoising=True):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    cycles = scene.cycles
    cycles.samples = samples
    cycles.use_denoising = use_denoising
    cycles.feature_set = 'EXPERIMENTAL'
    cycles.device = 'GPU'
    
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    cycles_prefs.get_devices()
    
    for compute_type in ['OPTIX', 'CUDA', 'HIP', 'ONEAPI', 'METAL']:
        try:
            cycles_prefs.compute_device_type = compute_type
            cycles_prefs.get_devices()
            
            devices_enabled = False
            for device in cycles_prefs.devices:
                if device.type == compute_type or device.type == 'CPU':
                    device.use = True
                    devices_enabled = True
            
            if devices_enabled:
                break
        except:
            continue
    
    scene.render.resolution_x = 780
    scene.render.resolution_y = 780
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    
    return scene


def render_all_cameras(output_dir, filename_prefix="render", cameras=None):
    if cameras is None:
        cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    
    os.makedirs(output_dir, exist_ok=True)
    output_files = []
    
    for camera in cameras:
        bpy.context.scene.camera = camera
        output_path = os.path.join(output_dir, f"{filename_prefix}_{camera.name}.png")
        bpy.context.scene.render.filepath = output_path
        
        print(f"Rendering from {camera.name}...")
        bpy.ops.render.render(write_still=True)
        
        output_files.append(output_path)
    
    return output_files


def main(format_type, furniture_data, object_sizes=None, normalize=True, 
         room_size=None, clear=True, create_room=True,
         setup_cameras_flag=True, setup_lighting_flag=True, setup_render=False, 
         render_output_dir=None, mesh_files=None, match_threshold=40):
    if clear:
        clear_scene(keep_camera=False, keep_lights=False)
    
    if format_type == 4 or (format_type is None and detect_holodeck_format(furniture_data)):
        format_type = 4
    
    if format_type == 1:
        obj_types = list(set(name.rsplit('-', 1)[0] for name in furniture_data.keys()))
    elif format_type == 2:
        obj_types = list(set(item[0] for item in furniture_data[0]['object_list']))
    elif format_type == 3:
        obj_types = list(set(
            obj['label'].rsplit('_', 1)[0] 
            for obj in furniture_data['objects']
            if 'ego' not in obj['label'] and not (obj['bbox'][2] == 0 and obj['bbox'][3] == 0)
        ))
    elif format_type == 4:
        obj_list = furniture_data.get('floor_objects', furniture_data.get('objects', []))
        obj_types = list(set(
            obj.get('object_name', obj.get('id', '').split(' ')[0]).rsplit('-', 1)[0]
            for obj in obj_list
            if obj.get('object_name') or obj.get('id')
        ))
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    resolved_mesh_files = build_mesh_mapping(
        obj_types, MESH_DIR, mesh_files, match_threshold
    )
    
    if format_type == 1:
        if object_sizes is None:
            print("Extracting object sizes from mesh files...")
            object_sizes = extract_mesh_dimensions(MESH_DIR, resolved_mesh_files)
        
        parsed = parse_format1(furniture_data, object_sizes=object_sizes, 
                               mesh_dir=MESH_DIR, mesh_files=resolved_mesh_files)
        
        if room_size is None:
            room_size = infer_room_size_format1(furniture_data, object_sizes)
        
    elif format_type == 2:
        parsed = parse_format2(furniture_data)
        
        if room_size is None:
            room_size = infer_room_size_format2(furniture_data)
        
    elif format_type == 3:
        if room_size is None:
            room_size = infer_room_size_format3(furniture_data)
        
        if object_sizes is None:
            object_sizes = extract_mesh_dimensions(MESH_DIR, resolved_mesh_files)
        
        parsed = parse_format3(furniture_data, normalize=normalize, room_size=room_size)
    
    elif format_type == 4:
        if room_size is None:
            room_size = infer_room_size_format4(furniture_data)
        
        parsed = parse_format4(furniture_data, room_size=room_size)
    
    room_size = normalize_room_size(room_size)
    room_width, room_depth = room_size
    
    print(f"Parsed {len(parsed)} objects from format {format_type}")
    print(f"Room size: {room_width:.2f} x {room_depth:.2f}")
    
    if create_room:
        create_floor(room_size=room_size, color_hex=FLOOR_COLOR_HEX)
    
    placed = place_all_furniture(parsed, MESH_DIR, resolved_mesh_files, 
                                 scale_to_fit=True, match_threshold=match_threshold)
    print(f"Successfully placed {len(placed)} objects")
    
    lights = []
    if setup_lighting_flag:
        lights = setup_lighting(room_size=room_size)
    
    cameras = []
    if setup_cameras_flag:
        cameras = setup_cameras(room_size=room_size)
    
    if setup_render:
        setup_cycles_gpu(samples=128, use_denoising=True)
    
    render_paths = []
    if render_output_dir:
        render_paths = render_all_cameras(render_output_dir, 
                                          filename_prefix="scene",
                                          cameras=cameras)
    
    return {
        'placed_objects': placed,
        'cameras': cameras,
        'lights': lights,
        'render_paths': render_paths,
        'room_size': room_size,
        'mesh_mapping': resolved_mesh_files,
    }        


import sys
import argparse

if __name__ == "__main__":

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--format", type=int, required=True)
    parser.add_argument("--json", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--room-size", nargs=2, type=float, default=None)
    args = parser.parse_args(argv)

    with open(args.json, "r") as f:
        furniture_data = json.load(f)

    room_size = tuple(args.room_size) if args.room_size else None

    print(args.format, furniture_data, args.normalize, room_size, args.output)

    main(
        format_type=args.format,
        furniture_data=furniture_data,
        normalize=args.normalize,
        room_size=room_size,
        render_output_dir=args.output,
    )