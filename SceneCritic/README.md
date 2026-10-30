> Colab setup: see `notebooks/SceneCritic_Colab.ipynb` or the combined repository's root notebook. `requirements.txt` pins direct dependencies from the local dspy environment; `reproducibility/requirements-local-py310.txt` records its full package snapshot. `.env` is private; `.env.example` lists settings. The original instructions below describe the upstream pipeline, including separate Blender/data requirements.

# SceneCritic

Official repository for SceneCritic: A Symbolic Evaluator for 3D Indoor Scene Synthesis

## Features

- Generation of the **SceneOnto** dataset and its structure
- 3D scene generation using our testbed and other pre-existing methods
- Rendering of generated 3D scene layouts
- Evaluation using **SceneCritic**

## Repository Structure

- `ontology` : SceneOnto dataset generation
- `testbed` : Scene layout generation code
- `benchmark` : Benchmark example for our testbed
- `render` : Rendering script
- `SceneCritic` : SceneCritic evaluation

## Clone

Clone this repository before installing dependencies.

```bash
git clone https://github.com/DIASENGUPTA/SceneCritic.git
cd SceneCritic
```

## Installations

### Create Conda Environment

```bash
conda create --name dspy python=3.10
conda activate dspy
```

### Install Core Dependencies

```bash
pip install dspy
pip install pydantic
```

### Install other dependencies

```bash
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

pip install matplotlib

pip install opencv-python
```

### Install SGLang and FlashInfer (Local version run)

Only install this if you want to run your models locally.

```bash
pip3 install sgl-kernel

pip install https://github.com/flashinfer-ai/flashinfer/releases/download/v0.1.1/flashinfer-0.1.1+cu121torch2.1-cp310-cp310-linux_x86_64.whl

pip install --no-cache-dir "sglang[all]"
```

> **Note:** Ensure CUDA 12.1 is available on your system.

### Run Model Server (Local version run)

Use this local version or use model APIs from hugging-face or Vertex AI according to convenience for backbone.

```bash
CUDA_VISIBLE_DEVICES=2 python -m sglang.launch_server \
  --port 7501 \
  --model-path <model_name>
```

## Ontology Data Generation

```bash
cd ontology
python generate_ontology.py \
    --threedfront_dir /path/to/3D-FRONT \
    --model_info /path/to/model_info.json \
    --scannet_dir /path/to/ScanNet \
    --vg_dir /path/to/VisualGenome \
    --output_dir ./ontology_output
```

model_info.json is located inside 3D-FRONT/3D-FUTURE-model.

### Ontology Directory Structure

The `ontology_output/` directory contains the generated ontology statistics and verifier configuration files for different room types.

```bash
ontology_output/
├── mining_report.txt
├── object_ontology_summary.csv
├── verifier_config_bedroom.json
├── verifier_config_bookstore.json
├── verifier_config_buffet_restaurant.json
├── verifier_config_classroom.json
├── verifier_config_computer_room.json
├── verifier_config_dining_room.json
├── verifier_config_living_room.json
...
```

- **`mining_report.txt`**: Contains logs and summary of the ontology extraction process.
- **`object_ontology_summary.csv`**: Provides statistical summaries (dimensions, co-occurrence, etc.) for all object categories.
- **`verifier_config_*.json`**: Room-specific configuration verification files used by the evaluation framework to verify scene plausibility based on learned ontology priors.

Our version of clean dataset can be accessed here:

🔗 [Download SceneCritic Data](https://drive.google.com/file/d/1Rj7WVNzUnkVp-GFBg4KILXGlTlJJxpkQ/view?usp=sharing)

## SceneCritic testbed

### Benchmark structure

These are some examples of our benchmark

```bash
benchmark_base_rooms/
├── bedroom/
│ ├── bedroom_1.json
│ ├── bedroom_2.json
│ ├── bedroom_3.json
├── bookstore/
├── buffet_restaurant/
├── classroom/
├── computer_room/
├── dining_room/
├── living_room/
```

### Run testbed

```bash
chmod +x testbed/run_layout_ours_varroom.sh
./testbed/run_layout_ours_varroom.sh benchmark/benchmark_base_rooms scene_jsons Heuristic_refinement.py
```

This script will generate scene jsons from our testbed setup. You can extract scene jsons from any scene generation pipeline (in their own format) and use that for the following evaluation steps.

## Render 3D Scenes

Rendering our testbed generated scene jsons

```bash
blender --background --python render/render_for_scripts.py -- \
--format 1 \
--json scene.json \
--output renders \
--normalize \
--room-size 10 10
```

Currently, the render code can handle four repositories LayoutGPT, LayoutVLM, Holodeck and our testbed. Add your custom generated scene formats for generating renders. The path to mesh directory is a directory with objaverse assets renamed with their object names.

## Evaluate with SceneCritic

### Evaluator Input Format

```bash
scene_jsons/
├── Heuristic/
│ ├── <Critic method output 1> # Scene jsons for each refinement method(Heuristic) + Backbone
│ ├── <Critic method output 2>
│ └── ...
├── Image/
├── Img_Text/
├── LLM/
├── Sem_Text/
└── Other models/
```


### Run SceneCritic Evaluator

```bash
blender -b --python mesh_dimension.py

chmod +x ./SceneCritic/SceneCritic_evaluator.sh
./SceneCritic/SceneCritic_evaluator.sh scene_jsons scene_jsons/SceneCritic_output mesh_dimension.json
```

### Evaluator Output Format

```bash
scene_jsons/SceneCritic_evaluator/
├── Heuristic/ # Individual evaluation of each method
├── Image/
├── Img_Text/
├── LLM/
├── Sem_Text/
├── Other models/
├── run.log
└── summary_all.tsv
```

### Aggregate scene_jsons results

```bash
python SceneCritic/aggregate_SceneCritic.py SceneCritic/SceneCritic_output/summary_all.tsv
```

Adopting SceneCritic to other method generated jsons will need a function to map the json to SceneCritic format, followed by verifying the orientation information convention(+ x axis vs placement direction convention) and modifying the SceneCritic Orientation Verifier accordingly. I have added example of modification made for `LayoutVLM` as `SceneCritic_layoutvlm.py` for reference.
