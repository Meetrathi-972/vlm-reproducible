import bpy
import json
import os
from pathlib import Path
from mathutils import Vector

MESH_DIR = "/path/to/mesh_directory" #Same mesh directory as used for rendering
OUTPUT_FILE = "mesh_dimensions.json"
MESH_EXTENSIONS = {'.glb', '.gltf', '.obj', '.fbx'}

def import_mesh(filepath):
    ext = Path(filepath).suffix.lower()
    existing_objects = set(bpy.data.objects)
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.obj':
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    return list(set(bpy.data.objects) - existing_objects)

dims = {}
for filename in os.listdir(MESH_DIR):
    ext = Path(filename).suffix.lower()
    if ext not in MESH_EXTENSIONS:
        continue
    filepath = os.path.join(MESH_DIR, filename)
    base_name = Path(filename).stem.lower()

    # Clear scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    imported = import_mesh(filepath)
    if not imported:
        continue

    min_coords = [float('inf')] * 3
    max_coords = [float('-inf')] * 3
    for obj in imported:
        if obj.type == 'MESH':
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ Vector(corner)
                for i in range(3):
                    min_coords[i] = min(min_coords[i], world_corner[i])
                    max_coords[i] = max(max_coords[i], world_corner[i])

    if min_coords[0] != float('inf'):
        dims[base_name] = [
            max_coords[0] - min_coords[0],
            max_coords[1] - min_coords[1],
            max_coords[2] - min_coords[2],
        ]

with open(OUTPUT_FILE, 'w') as f:
    json.dump(dims, f, indent=2)
print(f"Exported {len(dims)} mesh dimensions to {OUTPUT_FILE}")