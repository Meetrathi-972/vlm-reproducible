"""Run one benchmark room in a fresh directory and record its environment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room', type=Path, default=ROOT / 'benchmark/benchmark_base_rooms/bedroom/bedroom_1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONHASHSEED=str(args.seed), SCENECRITIC_SEED=str(args.seed), MPLBACKEND='Agg')
    room = json.loads(args.room.read_text())
    dims = room['room_dimension']
    # Use the existing interactive entrypoint with deterministic scripted input.
    fields = [str(args.output.resolve()), room['task_description'],
              ', '.join(f'{v} {k}' for k, v in room['assets'].items()),
              str(dims['room_dimension_x']), str(dims['room_dimension_y']), 'quit']
    if any('\n' in field for field in fields):
        raise ValueError('Room values must be single-line strings')
    manifest = {'seed': args.seed, 'python': sys.version,
                'input_sha256': hashlib.sha256(args.room.read_bytes()).hexdigest(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT)),
                'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'], text=True)}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    result = subprocess.run([sys.executable, '-m', 'testbed.Heuristic_refinement_qwen38'],
                            cwd=ROOT, env=env, input='\n'.join(fields) + '\n', text=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
