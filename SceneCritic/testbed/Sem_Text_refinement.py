"""
SceneCritic testbed with VLM as a Critic for refinement(Semantic+Text)
"""

import os
import sys
import re
import copy
import json
import os
import random
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import dspy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from PIL import Image
from pydantic import BaseModel, Field, conlist

import testbed.plot_utilities as plot_utilities
import testbed.utils as utils


def clean_model_output(raw_text: str) -> str:
    text = re.sub(r"```(?:json)?", "", raw_text, flags=re.I)

    text = re.sub(r"^(system|user|assistant|You are a helpful assistant).*", "", text, flags=re.I | re.MULTILINE)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("⚠️", "Do not", "Perform")):
            continue
        lines.append(line)

    return "\n".join(lines)


def extract_json(raw_text: str) -> Dict[str, Any]:
    cleaned = clean_model_output(raw_text)

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and any(k in data for k in ("atomic_tasks", "objects", "scene")):
            return data
    except json.JSONDecodeError:
        pass

    stack = []
    start_idx = None
    for i, c in enumerate(cleaned):
        if c == "{":
            if not stack:
                start_idx = i
            stack.append("{")
        elif c == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidate = cleaned[start_idx:i + 1]
                    try:
                        data = json.loads(candidate)
                        if any(k in data for k in ("atomic_tasks", "objects", "scene")):
                            return data
                    except json.JSONDecodeError:
                        continue
    return {}

lm = dspy.LM(
    "openai/TIGER-Lab/VL-Reasoner-72B",
    api_base="http://127.0.0.1:7501/v1",
    api_key="local",
    model_type="chat",
)

dspy.settings.configure(lm=lm)

class SceneObject(BaseModel):
    label: str = Field(..., description="Name of the object (e.g. 'chair').")
    object_type: str = Field(..., description="Choose 'bulk' or 'ornaments'.")
    anchor: Optional[str] = Field(None, description="Reference object for spatial relation, if any.")
    bbox: List[float] = Field(..., min_length=4, max_length=4,
                            description="Bounding box [x,y,w,h] in scene coordinates in meters.")
    position: List[float] = Field(..., min_length=2, max_length=2,
                                description="Object center [x+w/2,y+h/2] in scene coordinates in meters.")
    rotation_angle: Optional[float] = Field(0.0, description="Object rotation in degrees.")
    description: str = Field(..., description="Natural language description of the object.")
    concise_description: str = Field(..., description="Natural language description of only the object without mentioning its relation with other objects.")

class OutputSchema(BaseModel):
    add: Optional[List[SceneObject]] = Field(default_factory=list,
                                             description="Objects to add to the scene.")
    remove: Optional[List[str]] = Field(default_factory=list,
                                        description="Labels of objects to remove.")
    update: Optional[List[SceneObject]] = Field(default_factory=list,
                                                description="Objects to update (position, bbox, etc.).")

class AtomicTask(BaseModel):
    id: str = Field(..., description="Unique identifier for the task.")
    object_type: str = Field(..., description="Choose 'bulk' or 'ornaments'.")
    type: str = Field(..., description="Task type: 'Add', 'Remove', or 'Update'.")
    label: str = Field(..., description="Label of the object this task refers to.")
    anchor: Optional[str] = Field(None, description="Reference object for spatial relation, if any.")
    spatial_hint: Optional[str] = Field(None, description="Human-readable spatial instruction.")
    bbox_hint: Optional[List[int]] = Field(None, description="Optional bounding box hint [x,y,w,h].")
    output_schema: OutputSchema = Field(..., description="Structured delta to apply to the scene graph.")
    execution_prompt: str = Field(..., description="Concise LLM instruction to run this task.")



class SceneDescriptionGenerator(dspy.Signature):
    """
    Generate a concise updated description of the current scene
    using prior context, environment notes, and predicted area type.
    """
    random_seed: int = dspy.InputField(desc="Random seed for generating different output everytime.")
    previous_scene_description: str = dspy.InputField(
        desc="Textual description of the previous scene for contextual continuity."
    ) 
    env_note: str = dspy.InputField(
        desc="Environmental note or navigation instruction that guides how the current scene changes from the previous one."
    )

    new_scene_description: str = dspy.OutputField(
        desc="A short, coherent textual description of the new scene "
             "based on the provided context and area type."
    )
    concise_scene_name: str = dspy.OutputField(
        desc="One word name of the area you are generating."
    )
    scene_area: str = dspy.OutputField(
        desc="Indicate whether this is an indoor/outdoor area."
    )

class ObjectChoicePlanner(dspy.Signature):
    """Decide potential bulk and ornament objects for the described scene."""
    random_seed: int = dspy.InputField(desc="Random seed for generating different output everytime.")
    env_note: str = dspy.InputField(
        desc="Environmental note or navigation instruction that guides how the current scene changes from the previous one with some examples of object choice."
    )
    scene_description: str = dspy.InputField(
        desc="Scene description for the current scene."
    )
    choice_restriction_prompt:str = dspy.InputField(
        desc="Explicitly mentions rules for object choice."
    )
    bulk_objects: List[str] = dspy.OutputField(
        desc="Suggest a list of large, structural, or fixed objects (e.g., building, tree, table, sofa, road) that can be present in this scene."
    )
    ornament_objects: List[str] = dspy.OutputField(
        desc="Suggest a list of smaller, decorative, or movable objects (e.g., lamp, plant, trash bin, signboard) that can be present in this scene."
    )

class AtomicTaskPlanning(dspy.Signature):
    random_seed: int = dspy.InputField(desc="Random seed for generating different output everytime.")
    planning_instruction_prompt: str = dspy.InputField(
        desc="Instruction describing how to update a scene as you navigate to new areas, and"
         "decompose it into atomic tasks."
    )
    atomic_tasks_json: List[AtomicTask] = dspy.OutputField(
        desc="List of atomic tasks. Output must be a JSON."
    )

class ObjectPlacementGenerator(dspy.Signature):
    random_seed: int = dspy.InputField(desc="Random seed for generating different output everytime.")
    instruction_prompt: str = dspy.InputField(desc="Prompt describing object placement plan and instructions to place.")
    object_placement_feedback: str = dspy.InputField(
        desc="Optional: feedback from the previous object placement refinement iteration."
    )
    placement_json: List[SceneObject] = dspy.OutputField(desc="The placement output should be in strict json format.")

class AtomicPlanningSelfCheck(dspy.Signature):
    check_prompt: str = dspy.InputField(
        desc="Planning text describing atomic tasks for scene or asset placement that needs to be reviewed along with the plan generation instruction."
    )
    logical_object_grouping: str = dspy.OutputField(
        desc="Answer Yes or No: Are semantically related assets grouped or sequenced logically?"
    )
    object_completeness: str = dspy.OutputField(
        desc="Answer Yes or No: Are all objects listed under \u201cObjects to be placed\u201d in the Planning Instruction included in the final placement, with no additional or missing objects?"
    )
    verdict: str = dspy.OutputField(
        desc="Provide a concise verdict summarizing any detected issues and suggestions for improvement."
    )


class ScenePlacementSelfCheck(dspy.Signature):
    check_prompt: str = dspy.InputField(
        desc="Scene placement description and rules to review."
    )

    semantic_plot: Optional[dspy.Image] = dspy.InputField(
        desc="Top semantic view of the scene. Refer to this for critiquing the placement."
    )
    overlap_check: str = dspy.OutputField(
        desc="Answer Yes or No: None of the objects have any overlapping bounding box right?"
    )
    area_check: str = dspy.OutputField(
        desc="Answer Yes or No: Are all objects inside the specified room area?"
    )
    rule_check: str = dspy.OutputField(
        desc="Answer Yes or No: Are all general placement rules (alignment, support, spacing) followed?"
    )
    dimension_check: str = dspy.OutputField(
        desc="Answer Yes or No: Are all object dimensions correct?"
    )
    orientation_check: str = dspy.OutputField(
        desc="Answer Yes or No: Are all objects correctly oriented?"
    )
    verdict: str = dspy.OutputField(
        desc="Provide a detailed feedback about the questions asked in overlap_check, area_check, rule_check, dimension_check and orientation_check for better placement and orientation of the objects in room to make it visually more appealing."
    )

class OrientationRefiner(dspy.Signature):
    """
    Refine the rotation_angle of each object using realistic spatial alignment rules.
    Input and output are lists of SceneObject JSONs (same as ObjectPlacementGenerator).
    """
    random_seed: int = dspy.InputField(
        desc="Random seed for generating different output everytime."
    )
    orientation_prompt: str = dspy.InputField(
        desc="Instructions to improve orientation of each object relative to the current scene and other placed objects."
    )
    placement_json: List[SceneObject] = dspy.InputField(
        desc="List of SceneObject dicts to be refined."
    )
    scene_side_view_1: Optional[dspy.Image] = dspy.InputField(
        desc="One Side view of the scene. Refer to this for critiquing the placement."
    )
    scene_side_view_2: Optional[dspy.Image] = dspy.InputField(
        desc="Another Side view of the scene. Refer to this for critiquing the placement."
    )
    scene_front_view: Optional[dspy.Image] = dspy.InputField(
        desc="Front view of the scene. Refer to this for critiquing the placement."
    )
    scene_back_view: Optional[dspy.Image] = dspy.InputField(
        desc="Back view of the scene. Refer to this for critiquing the placement."
    )
    scene_top_view: Optional[dspy.Image] = dspy.InputField(
        desc="Top view of the scene. Refer to this for critiquing the placement."
    )
    updated_scene_json: List[SceneObject] = dspy.OutputField(
        desc="List of SceneObject dicts with updated rotation_angle fields only."
    )


class ScaleRefiner(dspy.Signature):
    """
    Refine the scale of each object using semantic and spatial reasoning.
    Input and output are lists of SceneObject JSONs (same as ObjectPlacementGenerator).
    """
    random_seed: int = dspy.InputField(
        desc="Random seed for generating different output everytime."
    )
    scale_prompt: str = dspy.InputField(
        desc="Prompt describing current scene and instructions to improve bounding box scales."
    )
    placement_json: List[SceneObject] = dspy.InputField(
        desc="List of SceneObject dicts to be refined."
    )
    updated_scene_json: List[SceneObject] = dspy.OutputField(
        desc="List of SceneObject dicts with updated bbox and position fields only."
    )



def build_planning_prompt(position, env_note, prev_scenegraphs, save_dir, dim_x, dim_y, dx, dy, object_store, scene_desc = None, ornaments_included = False):
    x, y = position
    role_block = f"Role: You are an expert layout planner.\n"

    env_block = f"""
    Environment note: {env_note}
    """

    placement_order = f"""
    - Objects to be placed: {object_store}. Do not place any other objects than this list.
    - DO NOT PLACE ANY ORNAMENTS.
    - You have to place all the objects mentioned.
    """

    object_check = f"""
    Strictly follow these rules:
    - Objects not to include: 
        - windows, doors, walls, partitions or anything attached to it.
        - floors, roofs.
        - roads, street, pavement, footpath, sidewalk, pathway, lawn and any other part of ground.
        - objects like counter that has multiple english meanings.
    - Include only specific objects, not generalized names like new_object, appliances etc.
    - Do not include any objects in plural.
    """

    prev_objects = []
    for sg in prev_scenegraphs:
        if isinstance(sg, dict):
            if "scene" in sg and isinstance(sg["scene"], dict):
                prev_objects.extend(sg["scene"].get("objects", []))
            else:
                prev_objects.extend(sg.get("objects", []))

    prev_objects_summary = ", ".join([
        f"{obj.get('label','unknown')} at {obj.get('position',[0,0])}"
        for obj in prev_objects if obj.get("label") != "ego_person"
    ])

    if prev_objects:
        if dy != 0:
            dire = f"move along Y-axis by {dim_y} meters"
            y = [position[1], position[1] + dim_y]
            x = [position[0] - dim_x//2, position[0] + dim_x//2]
            new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
        elif dx != 0:
            dire = f"move along X-axis by {dim_x} meters"
            y = [position[1], position[1] + dim_y]
            x = [position[0] - dim_x//2, position[0] + dim_x//2]
            new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
        else:
            dire = f"stay at same"
            new_area_range = None

        movement_block = f"""
    Previous scene objects: {prev_objects_summary}
    Movement Note: You {dire} to absolute ({position}).
    """
        consistency_block = f"""
    Consistency Note:
    - Objects mentioned in instruction should appear.
    """
    else:
        x = [position[0]-dim_x//2,position[0]+dim_x//2]
        y = [position[1],position[1] + dim_y]
        new_area_range = f"x ranging between {x[0]} meters and {x[1]} meters, y ranging between {y[0]} meters and {y[1]} meters"
        movement_block = f"Movement Note: You are at {position}."
        consistency_block = f"""
    Consistency Note:
    - Initialize scene with new objects ONLY in the visible area: {new_area_range}.
    - Generate a scene description.
    """

    planning_prompt = role_block + env_block +  placement_order + object_check

    if scene_desc:
        scene_desc_block = f"""
        New Scene Description:
        {scene_desc}
        """
        planning_prompt = planning_prompt + scene_desc_block

    semantic_block = f"""
    Object Arrangement and Spatial Relationships:
    - Objects should be arranged naturally as seen in real world environment (e.g., chairs around table, cars parked along street, trees spaced in a park, lamp on top of table/desk, pillow on sofa/bed , television facing sofa , vase centered on table ,  bird feeders hanging on a tree, trash bins beside bench ).
    - Define clear spatial relationships between objects (e.g., next to, under, near, on top of, centered on, beside).
    - Align most larger objects along area boundaries(e.g. walls, fences or paths) and corners to reflect natural real-world arrangements and maximise navigable spaces.
    - Ensure bounding boxes and positions are consistent with the defined spatial relationships.
    
    Follow semantic asset grouping:
    - A semantic asset group is a collection of assets that are logically, functionally, or spatially related within a scene. Assets in a group support a common purpose or activity and together form a meaningful subscene. 
    - Group assets based on function (used together), semantics (conceptually related), or geometry (close in space or aligned).
    - Examples:
        - Sleeping area: Bed + Nightstand + Lamp (used together for resting and reading)
        - Entertainment area: Sofa + TV Console + TV  (functionally related for watching)
        - Work area: Desk + Chair + Laptop (supports the working activity)
        - Dining area: Dining Table + Chairs + Chandelier (forms a cohesive eating zone)
    """
    
        
    task_block = """
Task:
Decompose this instruction into ATOMIC TASKS for the design process. The output should ONLY be a valid json. Do not include position or bounding box in execution prompt.""" 
    planning_prompt = planning_prompt + semantic_block + task_block

    file_path = os.path.join(save_dir, f"planning_prompt_{position[0]}_{position[1]}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(planning_prompt)

    return planning_prompt, new_area_range, consistency_block, movement_block

def build_object_placement_prompt(plan_text, position, dim_x, dim_y, consistency_block, movement_block, env_note, new_area_range):
    role_block = f"Role: You are an expert real world planner.\n"
    env_block = f"Environment note: {env_note}.\n"
    return role_block + env_block + f"""

Plan summary for the room:
{plan_text}

Placement Rules:
- You are populating a {dim_x} meters * {dim_y} meters room.
- The room dimensions are x ranging between -{dim_x/2} to {dim_x/2} and y ranging between 0 to {dim_y}. Bounding boxes SHOULD NOT go out of the room area.
- Design such that two BULK objects bounding boxes SHOULD NOT INTERSECT each other unless the relationship is under or on top of.
- While placing, verify that the front of the object being placed and already placed objects remains clearly visible and is not obstructed by any larger object positioned in front of it.
- Maintain realistic distances between the objects you are placing to make the room visually appealing. Do not clutter them together.
- Align most larger objects along area boundaries(e.g. walls, fences or paths) to reflect natural real-world arrangements and maximise navigable spaces.
- Each object must include: label(same as the name mentioned in plan, do not add any adjective), object_type, anchor, description, bbox [x,y,w,h] where (x+w/2,y+h/2) is the position/center, (x,y) is the bottom left corner and (w,h) is the width and height, description, position [x+w/2,y+h/2], rotation angle.
- Object sizes and bounding boxes must be semantically consistent with other objects in the scene and the room dimension.
- Positions and bounding boxes must accurately reflect spatial relationships (e.g., next to, under, on top of, centered on, near).
- Anchored objects must be placed and oriented relative to their reference object according to the defined spatial relationship.

Instructions:
- Output ONLY valid JSON with fields: objects.
"""

def build_orientation_prompt(scene: Dict[str, Any]) -> str:
    objs = scene.get("objects", [])
    obj_lines = "\n".join([
        f"- {o['label']}: {o.get('description', '')} (anchor: {o.get('anchor', 'none')})"
        for o in objs
    ])

    return f"""
You are refining object orientations in a 3D indoor or outdoor scene.
Assign realistic rotation angles (in degrees, 0\u2013360) to each object.

Interpretation:
- 0\u00b0: facing forward (toward the ego or camera)
- 90\u00b0: facing right
- 180\u00b0: facing backward
- 270\u00b0: facing left

Guidelines:
- Objects attached to walls should face away from the wall.
- Sofas and chairs should face TVs or tables if present.
- Tables should align with nearby seating.
- TVs and monitors should face the main sitting area.
- Beds should face away from their headboard wall.
- Cabinets, wardrobes, and shelves should be against walls and face outward.
- Appliances (stoves, sinks, fridges) should face the open workspace or usable area.
- Floor lamps, indoor plants, and static decor items usually have 0\u00b0 rotation.
- Benches should face open spaces, walkways, or scenic views.
- Vehicles should face their direction of travel or toward open space.
- Trees, rocks, and natural outdoor elements should maintain 0\u00b0 rotation.
- Freestanding objects in open areas should face the scene center or main viewpoint.
- Groups of similar objects (e.g., multiple chairs) should be aligned symmetrically.
- Desks should face open space or toward the user's seating direction.
- Bathroom fixtures (sinks, toilets) should face the center or entry of the room, not walls.

Scene objects:
{scene}

- Output ONLY valid JSON with fields: objects exactly similar to input. Do not output reasoning.
""".strip()


def build_scale_prompt(scene: Dict[str, Any]) -> str:
    print(scene, "Scene build")
    objs = scene.get("objects", [])
    obj_labels = ", ".join([o["label"] for o in objs])

    return f"""
You are refining object scales in a realistic 3D scene.

Adjust the size of each object's bounding box based on its natural proportion relative to surrounding objects in the scene. Maintain realistic scale relationships as seen in real environments.
Ensure relative size consistency \u2014 smaller items (e.g., lamps, books, vases) should not exceed larger reference objects (e.g., tables, beds, sofas).
Examples:
- A chair should be smaller than a desk/table.
- A desk should be smaller than a bed.
- A nightstand should be lower than a bed.
- A sofa should be larger than a side table.
- Trees should be taller than nearby benches or cars.
- Buildings should dwarf trees and vehicles.
Keep proportions coherent within the environment (indoor or outdoor), adjusting bounding box dimensions for necessary objects so all objects appear physically plausible when viewed together.

Scene objects: 
{scene}

- Output ONLY valid JSON with fields: objects exactly similar to input.
""".strip()




class SceneDescription(dspy.Module):

    def __init__(self):
        super().__init__()
        self.scene_desc = dspy.ChainOfThought(SceneDescriptionGenerator)

    def forward(
        self,
        env_note: str,
        scene_desc: str
    ) -> str:

        print(f"\n Generating new scene description for position ")
        output = self.scene_desc(random_seed=utils.generate_random_seed(),
            previous_scene_description=scene_desc,
            env_note=env_note
        )
        new_scene_desc = getattr(output, "new_scene_description", "No scene generated.")
        new_scene_name = getattr(output, "concise_scene_name", "No scene generated.")
        new_scene_area = getattr(output, "scene_area", "No scene generated.")
        return new_scene_desc, new_scene_name, new_scene_area
    

class ObjectChoice(dspy.Module):
    def __init__(self):
        super().__init__()
        self.planner = dspy.ChainOfThought(ObjectChoicePlanner)

    def suggest_objects(self, env_note: str, scene_description: str) -> Dict[str, List[str]]:
        result = self.planner(random_seed=utils.generate_random_seed(),
            env_note=env_note,
            scene_description=scene_description,
            choice_restriction_prompt = f"""
            Strictly follow these rules:
- Objects not to include: 
    - windows, doors, walls, partitions or anything attached to it.
    - floors, roofs.
    - roads, street, pavement, footpath, sidewalk, pathway,lawn and any other part of ground.
    - objects like counter that has multiple english meanings.
- Include only specific objects, not generalized names like new_object, appliances etc.
- Do not include any objects in plural.
- Do not repeat any object more than twice. Try to avoid repeating objects.
            """
        )

        bulk = result.bulk_objects or []
        ornaments = result.ornament_objects or []

        return {"bulk_objects": bulk, "ornament_objects": ornaments}

class PlanAtomicTasks(dspy.Module):
    def __init__(self, save_dir):
        super().__init__()
        self.atomic_planner = dspy.ChainOfThought(AtomicTaskPlanning)
        self.planning_critic = dspy.ChainOfThought(AtomicPlanningSelfCheck)

        self.planner_refine = dspy.Refine(
            module=self.atomic_planner,
            N=5,
            reward_fn=self.planning_reward,
            threshold=1.0
        )
        self.save_dir = save_dir
        self._refine_logs =[]

    def forward(self, position, env_note, scene_desc, object_store, prev_scenegraphs, dx, dy):
        
        planning_prompt, new_area_range, consistency_note, movement_note = build_planning_prompt(position=position, env_note=env_note, prev_scenegraphs=prev_scenegraphs, save_dir = self.save_dir, dim_x = dim_x, dim_y = dim_y, dx = dx, dy=dy, object_store=object_store, scene_desc=scene_desc, ornaments_included = False)
        
        print(planning_prompt, "Planning Prompt")
        allowed_objs = utils._parse_allowed_objects(planning_prompt)

        pred = self.atomic_planner(random_seed=utils.generate_random_seed(), planning_instruction_prompt=planning_prompt)
        parsed_pred = pred.atomic_tasks_json
        if parsed_pred and isinstance(parsed_pred[0], AtomicTask):
            parsed_pred = [task.model_dump() for task in parsed_pred]
            parsed_pred = {"atomic_tasks": parsed_pred} 

        with open(os.path.join(self.save_dir, f"planning_{position[0]}_{position[1]}.json"), "w") as f:
            json.dump(parsed_pred, f, indent=2)
        return parsed_pred, new_area_range, consistency_note, movement_note, allowed_objs
    
    def planning_reward(self, args, pred):
        instruction = args.get("planning_instruction_prompt", "")
        planning_feedback = args.get("planning_feedback", "")
        print(f"Previous planning_feedback passed in:\n{planning_feedback}\n")

        position = args.get("position", (0,0))
        plan = getattr(pred, "atomic_tasks_json", pred)

        parsed_pred = plan
        if parsed_pred and isinstance(parsed_pred[0], AtomicTask):
            parsed_pred = [task.model_dump() for task in parsed_pred]
        parsed_pred = {"atomic_tasks": parsed_pred}
        save_path = os.path.join(
            self.save_dir,
            f"planning_{position[0]}_{position[1]}_pass{len(self._refine_logs)+1}.json"
        )
        with open(save_path, "w") as f:
            json.dump(parsed_pred, f, indent=2)
        print(f"[Saved plan JSON to: {save_path}]")

        planning_text = "\n".join(
            task.execution_prompt
            for task in plan
            if hasattr(task, "execution_prompt") and task.execution_prompt
        )

        check_prompt = (
            f"Planning instruction:\n{instruction}\n\n"
            f"Generated atomic plan:\n{planning_text}\n\n"
            "Now review the above atomic plan carefully and answer the following:\n"
            "1. Are semantically related assets grouped or sequenced logically? (Answer Yes or No)\n"
            "2. Are all objects listed under \u201cObjects to be placed\u201d in the Planning Instruction included in the final placement, with no additional or missing objects? (Answer Yes or No)\n\n"
            "After answering the above two questions, provide a **verdict** that:\n"
            "- Summarizes any detected issues.\n"
            "- Do not add any objects that are not listed under \u201cObjects to be placed\u201d in the Planning Instructions.\n"
            "- Suggest concrete improvements, modifications, or additions needed to make the plan fully correct and logically ordered.\n"
            "- If everything is perfect, clearly state that the plan is consistent and needs no changes."
        )
        print("\nPlanning Check Prompt\n", check_prompt)

        critic_pred = self.planning_critic(check_prompt=check_prompt)

        grouping_ans = critic_pred.logical_object_grouping.strip().lower().startswith("y")
        completeness_ans = critic_pred.object_completeness.strip().lower().startswith("y")

        if completeness_ans and grouping_ans:
            reward = 1.0
        elif completeness_ans:
            reward = 0.7
        elif grouping_ans:
            reward = 0.5
        else:
            reward = 0.2

        feedback_text = f"Planning Critic Feedback:\n{critic_pred.verdict}"
        args["planning_feedback"] = feedback_text

        self._refine_logs.append({
            "iteration": len(self._refine_logs) + 1,
            "planning_text": planning_text,
            "answers": {
                "logical_sequence": critic_pred.logical_object_grouping,
                "completeness": critic_pred.object_completeness,
            },
            "verdict": critic_pred.verdict,
            "reward": reward,
            "feedback_text": feedback_text
        })
        log_path = os.path.join(self.save_dir, f"refine_logs_planning_{position[0]}_{position[1]}.json")
        with open(log_path, "w") as f:
            json.dump(self._refine_logs, f, indent=4)

        print(reward, "Planning Reward")

        return reward




class ObjectPlacementManager(dspy.Module):
    def __init__(self, save_dir):
        super().__init__()
        self.placement_generator = dspy.ChainOfThought(ObjectPlacementGenerator)
        self.placement_critic = dspy.ChainOfThought(ScenePlacementSelfCheck)
        self.refine_placement_generator = dspy.Refine(
            module=self.placement_generator,
            N=5,
            reward_fn=self.placement_reward,
            threshold=1.0
        )
        self.save_dir = save_dir
        self._refine_logs =[]

    def forward(self, position, plan, new_area_range, allowed_objs, dim_x, dim_y, consistency_block, movement_block, base_scene, env_note="No note"):
        plan_text = "\n".join(
            task.get('execution_prompt', '') 
            for task in plan.get("atomic_tasks", []) 
            if task.get('execution_prompt')
        )
        prompt = build_object_placement_prompt( plan_text, position, dim_x, dim_y, consistency_block, movement_block, env_note, new_area_range)

        print(prompt,"Execution Plan")

        exec_pred = self.refine_placement_generator(random_seed=utils.generate_random_seed(), instruction_prompt=prompt, object_placement_feedback = "", position = position, dim_x=dim_x, dim_y=dim_y)
        delta = exec_pred.placement_json

        delta_objects = [utils.normalize_obj(o) for o in delta]

        if delta_objects:
            delta_objects = utils.assign_progressive_ids(
                delta_objects,
                scene_graph_dir=self.save_dir
            )
            
        base_scene["objects"].extend(delta_objects)

        return base_scene

    def placement_reward(self, args, pred):
        print("feedback",args.get("object_placement_feedback"))
        delta = getattr(pred, "placement_json", pred)
        delta_objects = [utils.normalize_obj(o) for o in delta]
        tmp_scene = {"objects": delta_objects}

        pos = args.get("position", (0, 0))
        save_dir = self.save_dir
        utils.visualize(tmp_scene, position=pos, save_dir=save_dir, save_name=f"check_{len(self._refine_logs) + 1}", sem_name=f"sem_plot_{len(self._refine_logs) + 1}")

        json_path = os.path.join(save_dir, f"check_{len(self._refine_logs) + 1}_{pos[0]}_{pos[1]}.json")

        for obj in tmp_scene.get("objects", []):
            if "rotation_angle" not in obj or obj["rotation_angle"] is None:
                obj["rotation_angle"] = 0.0

        with open(json_path, "w") as f:
            json.dump(tmp_scene, f, indent=2)
        print(f"[Saved delta JSON] {json_path}")

        placement_instruction = args.get("instruction_prompt", "")

        check_prompt = (
            f"Placement instruction:\n{placement_instruction}\n\n"
            f"Scene JSON:\n{delta}\n\n"
            "Now carefully review the above scene JSON and the rendered scene images from different views. "
            "Answer the following questions carefully and provide a detailed verdict.\n\n"
            "1. None of the objects have any overlapping bounding boxes, right? (Answer Yes or No)\n"
            "2. Are all objects placed inside the specified room area? (Answer Yes or No)\n"
            "3. Are all general placement rules (alignment, support, spacing) followed? (Answer Yes or No)\n"
            "4. Are all object dimensions correct? (Answer Yes or No)\n"
            "5. Are all objects correctly oriented? (Answer Yes or No)\n\n"
            "Provide a **verdict**:\n"
            "- Specify any placement or bounding box dimension changes related to the above 5 questions that could make the scene more visually appealing.\n"
            "- Identify any orientation adjustments needed to make the scene appear more realistic.\n"
            "- Describe any detected placement issues (overlaps, out-of-bounds, or rule violations).\n"
            "- Suggest clear, concrete adjustments or modifications to fix the problems.\n"
            "- If the scene placement is perfect, clearly state that all checks pass and no changes are required.\n"
        )

        blender_dir = os.path.join(save_dir,f"check_{len(self._refine_logs) + 1}_{pos[0]}_{pos[1]}")

        critic_pred = self.placement_critic(
        check_prompt=check_prompt, semantic_plot = dspy.Image(os.path.join(save_dir, f"sem_plot_{len(self._refine_logs) + 1}_{pos[0]}_{pos[1]}.png"))
    )

        reward = 0.0
        if critic_pred.overlap_check.strip().lower().startswith("y"):
            reward += 0.4
        if critic_pred.area_check.strip().lower().startswith("y"):
            reward += 0.2
        if critic_pred.rule_check.strip().lower().startswith("y"):
            reward += 0.2
        if critic_pred.dimension_check.strip().lower().startswith("y"):
            reward += 0.1
        if critic_pred.orientation_check.strip().lower().startswith("y"):
            reward += 0.1

        feedback_text = f"Placement Critic Feedback:\n{critic_pred.verdict}"
        args["object_placement_feedback"] = feedback_text

        self._refine_logs.append({
            "iteration": len(self._refine_logs) + 1,
            "scene_json": [utils.normalize_obj(o) for o in delta],
            "answers": {
                "overlap_check": critic_pred.overlap_check,
                "area_check": critic_pred.area_check,
                "rule_check": critic_pred.rule_check,
                "dimension_check": critic_pred.dimension_check,
                "orientation_check": critic_pred.orientation_check
            },
            "verdict": critic_pred.verdict,
            "reward": reward,
            "feedback_text": feedback_text
        })

        log_path = os.path.join(self.save_dir, f"refine_logs_placement_{pos[0]}_{pos[1]}.json")
        with open(log_path, "w") as f:
            json.dump(self._refine_logs, f, indent=4)
            

        print(reward, "Placement Reward")

        return reward




    

class OrientationRefinerModule(dspy.Module):
    def __init__(self, save_dir):
        super().__init__()
        self.refiner = dspy.ChainOfThought(OrientationRefiner)
        self.save_dir = save_dir

    def forward(self, scene: Dict[str, Any], pos, instruction) -> Dict[str, Any]:
        pass


class ScaleRefinerModule(dspy.Module):
    def __init__(self, save_dir):
        super().__init__()
        self.refiner = dspy.ChainOfThought(ScaleRefiner)
        self.save_dir = save_dir

    def forward(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        pass





class ScenePlanner(dspy.Module):
    def __init__(self, save_dir):
        super().__init__()
        self.scene_desc = SceneDescription()
        self.manager = ObjectPlacementManager(save_dir = save_dir) 
        self.object_choice = ObjectChoice()
        self.planner = PlanAtomicTasks(save_dir = save_dir)
        self.orientation_refiner = OrientationRefinerModule(save_dir)
        self.scale_refiner = ScaleRefinerModule(save_dir)
        self.save_dir = save_dir     
        

    def forward(self, position, prev_scenegraphs, instruction, objects_list, dim_x, dim_y, dx, dy, orientation = False, scale=False):

        env_note = f"Instruction: {instruction}" if instruction else "No special instruction."


        scene_desc, scene_name, scene_area = self.scene_desc(
            env_note=env_note,
            scene_desc=prev_scenegraphs[-1].get("scene_description", "") if len(prev_scenegraphs) != 0 else "No previous scene description!"
        )

        objects = self.object_choice.suggest_objects(env_note, scene_desc)

        plan, new_area_range, consistency_note, movement_note, allowed_objs = self.planner(position=position,
                                 env_note=env_note,
                                 scene_desc = None,
                                 object_store = objects_list,
                                 prev_scenegraphs=prev_scenegraphs,
                                 dx=dx, dy=dy)

        scene = self.manager(position, plan, new_area_range, allowed_objs, dim_x, dim_y, consistency_note, movement_note, base_scene = {"objects": []}, env_note=env_note)
        utils.visualize(scene, position = pos, save_dir = save_dir, save_name = "initial", sem_name = "sem_plot")
        
        if orientation:
            scene_tmp = self.orientation_refiner(scene, position, instruction)
            scene["objects"] = [o.model_dump() for o in scene_tmp if o.label != "ego_person"]
            utils.visualize(scene, position = pos, save_dir = save_dir, save_name = "orient")
        if scale:
            scene_tmp = self.scale_refiner(scene)
            scene["objects"] = [o.dict() for o in scene_tmp if o.label != "ego_person"]
            utils.visualize(scene, position = pos, save_dir = save_dir, save_name = "scale")
            

        scene["Environmental_note"] = env_note


        print(scene, "Scene")

        ego_obj = {
            "label": "ego_person",
            "object_type": "bulk",
            "anchor": None,
            "label_index": "ego_person",
            "position": [pos[0], pos[1]],
            "bbox": [pos[0], pos[1], 0, 0],
            "description": "Ego agent position",
            "rotation_angle": 0.0
        }
        scene["objects"].append(ego_obj)

        with open(os.path.join(self.save_dir, f"scene_pos_{position[0]}_{position[1]}.json"), "w") as f:
            json.dump(scene, f, indent=2)


        return scene

if __name__ == "__main__":

    save_dir = input("Enter the path to save/load scenegraphs: ").strip()
    os.makedirs(save_dir, exist_ok=True)

    scene_planner = ScenePlanner(save_dir = save_dir)
    
    all_prev_scenes = []

    pos = (0, 0)
    dim_x = None; dim_y = None

    prev_scene = {
    "objects": [],
    "scene_description": "No previous scene description.",
    }

    while True:
        instr = input("\nEnter environment instruction (or 'quit'): ").strip()
        if instr.lower() == "quit":
            break
        
        dx, dy = utils.extract_movement_from_note(instr)
        abs_x = pos[0] + dx * utils.STEP_SIZE
        abs_y = pos[1] + dy * utils.STEP_SIZE
        pos = (abs_x, abs_y)

        print("Ego moved to:", pos)

        scenegraph = utils.load_existing_scenegraph(pos, save_dir= save_dir)

        if scenegraph:
            print(f"Loaded existing scenegraph for position {pos}")
            result_scene = scenegraph
        else:
            obj_input = input("Enter a list of objects (comma-separated, e.g., 'chair, table, lamp'): ").strip()
            dim_x = int(input("x dimension of the room.").strip())
            dim_y = int(input("y dimension of the room.").strip())
            print("x dim", dim_x)
            print("y dim", dim_y)
            print("Asset", obj_input)

            result_scene = scene_planner(
                position=pos,
                prev_scenegraphs=all_prev_scenes,
                instruction=instr,
                objects_list=obj_input, 
                dim_x = dim_x,
                dim_y = dim_y, 
                dx=dx,
                dy=dy,
                orientation = False, scale=False
            )   
        

        utils.plot_scenegraph(result_scene, pos, save_dir=save_dir)
        print(f"Scene at position {pos} saved and plotted.")
        utils.visualize(result_scene, position = pos, save_dir = save_dir, save_name = "final", sem_name = "final_sem")
        prev_scene = result_scene
        all_prev_scenes.append(result_scene)