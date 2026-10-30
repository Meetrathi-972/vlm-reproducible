"""Standalone bedroom_1 fixed-placement API replay and top-down plotting.
Run with a Python environment containing matplotlib. No project imports.
The saved strict-greedy placement prompt is embedded below; upstream stages are frozen.
DeepInfra seed documentation: https://docs.deepinfra.com/api-reference/chat-completions/openai-chat-completions
"""
import argparse
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import urllib.request
import urllib.error
from datetime import datetime, timezone

PAYLOAD = {'model': 'Qwen/Qwen3.5-27B:deepinfra', 'messages': [{'role': 'system', 'content': 'You are a strict JSON generator. Output ONLY a single valid JSON value and nothing else. Do NOT include any reasoning, thinking, explanation, preamble, markdown, or code fences. Your entire response must be parseable by json.loads().'}, {'role': 'user', 'content': '\nRole: You are an expert real world planner.\nEnvironment note: Instruction: This is an indoor bedroom..\n\n\nPlan summary for the room:\nAdd a bed centered against the back wall.\nAdd a cabinet to the left of the bed.\nAdd a cabinet to the right of the bed.\nAdd a desk against the left wall.\nAdd a chair in front of the desk.\nAdd a shelf on top of the desk.\n\nPlacement Rules:\n- You are populating a 5 meters * 5 meters room.\n- The room dimensions are x ranging between -2.5 to 2.5 and y ranging between 0 to 5. Bounding boxes SHOULD NOT go out of the room area.\n- Design such that two BULK objects bounding boxes SHOULD NOT INTERSECT each other unless the relationship is under or on top of.\n- While placing, verify that the front of the object being placed and already placed objects remains clearly visible and is not obstructed by any larger object positioned in front of it.\n- Maintain realistic distances between the objects you are placing to make the room visually appealing. Do not clutter them together.\n- Align most larger objects along area boundaries(e.g. walls, fences or paths) to reflect natural real-world arrangements and maximise navigable spaces.\n- Each object must include: label(same as the name mentioned in plan, do not add any adjective), object_type, anchor, description, bbox [x,y,w,h] where (x+w/2,y+h/2) is the position/center, (x,y) is the bottom left corner and (w,h) is the width and height, description, position [x+w/2,y+h/2], rotation angle.\n- Object sizes and bounding boxes must be semantically consistent with other objects in the scene and the room dimension.\n- Positions and bounding boxes must accurately reflect spatial relationships (e.g., next to, under, on top of, centered on, near).\n- Anchored objects must be placed and oriented relative to their reference object according to the defined spatial relationship.\n\nInstructions:\n- Output ONLY valid JSON with fields: objects.\n\n\nOutput ONLY a JSON object of exactly this shape (no markdown, no commentary):\n{\n  "objects": [\n    {\n      "label": "<object name, same as plan, NO adjectives, singular>",\n      "object_type": "<\'bulk\' or \'ornaments\'>",\n      "anchor": "<reference object label for a spatial relation, or null>",\n      "bbox": [x, y, w, h],\n      "position": [x + w/2, y + h/2],\n      "rotation_angle": <float degrees>,\n      "description": "<natural language description, may mention relations>",\n      "concise_description": "<description of ONLY this object, no relations>"\n    }\n  ]\n}\nEvery object MUST include all of these fields.'}], 'seed': 1234, 'temperature': 0.0, 'top_p': 1.0, 'top_k': 1, 'max_tokens': 8192, 'frequency_penalty': 0.0, 'presence_penalty': 0.0, 'repetition_penalty': 1.0, 'stop': [], 'reasoning_effort': 'none', 'response_format': {'type': 'json_object'}, 'stream': False, 'n': 1}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def token_from_env(root):
    # Prefer the current saved SceneCritic token, avoiding stale inherited tokens.
    p=root/'SceneCritic/.env'
    if p.exists():
        for line in p.read_text().splitlines():
            m=re.match(r"^\s*(?:export\s+)?HF_TOKEN\s*=\s*(.*?)\s*$",line)
            if m and m[1].strip():
                return m[1].strip().strip('"\'')
    token=os.environ.get('HF_TOKEN')
    if not token: raise RuntimeError('No HF_TOKEN found')
    return token


def plot(ax, scene, title):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-2.5,0),5,5,fill=False,color='black',linewidth=2))
    counts={}
    for i,obj in enumerate(scene['objects']):
        x,y,w,h=map(float,obj['bbox'])
        label=obj['label']; counts[label]=counts.get(label,0)+1
        color=f'C{i%10}'
        ax.add_patch(Rectangle((x,y),w,h,facecolor=color,edgecolor=color,alpha=.3))
        cx,cy=x+w/2,y+h/2
        ax.text(cx,cy,f'{label} #{counts[label]}',ha='center',va='center',fontsize=8)
        theta=math.radians(float(obj.get('rotation_angle',0)))
        ax.arrow(cx,cy,.3*math.sin(theta),-.3*math.cos(theta),width=.015,color=color)
    ax.set(xlim=(-3.5,3.5),ylim=(-1,6),xlabel='x (m)',ylabel='y (m)',title=title)
    ax.set_aspect('equal');ax.grid(alpha=.25)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replays',type=int,default=10)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 2<=args.replays<=10:parser.error('--replays must be 2 through 10')
    root=Path(__file__).resolve().parents[2]
    out=args.output or Path(__file__).resolve().parent/'outputs'/('bedroom_1_standalone_replay_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    body=json.dumps(PAYLOAD,sort_keys=True,allow_nan=False).encode('utf-8')
    (out/'exact_request.json').write_bytes(body)
    report=dict(status='running',benchmark='bedroom_1',stage='placement',requested_replays=args.replays,completed_replays=0,api_attempts=0,retries=0,api_calls_other_stages=0,request_sha256=digest(body),generation_parameters={k:v for k,v in PAYLOAD.items() if k!='messages'},runs=[],comparison='Complete content UTF-8 bytes: no trimming, normalization, or reasoning removal; IDs/usage excluded',do_sample='Omitted: not documented in DeepInfra chat-completion schema',documentation='https://docs.deepinfra.com/api-reference/chat-completions/openai-chat-completions',limitation='Fixed saved placement prompt; not a replay of the complete multistage pipeline. Internal provider execution is not observable.',plot_convention='Axis-aligned bbox footprints, as in SceneCritic top-down plots; arrows use yaw 0=-y, 90=+x. Rendering does not modify comparison content.')
    contents=[];scenes=[]
    try:
        token=token_from_env(root)
        for i in range(1,args.replays+1):
            d=out/f'replay_{i:02d}';d.mkdir()
            # Send exactly these same bytes on every call. No retries or caching.
            (d/'request.json').write_bytes(body)
            request=urllib.request.Request('https://router.huggingface.co/v1/chat/completions',data=body,headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
            report['api_attempts']+=1
            print(f'Replay {i}/{args.replays}',flush=True)
            with urllib.request.urlopen(request,timeout=240) as response:
                raw=response.read()
            if token.encode() in raw:raise RuntimeError('Credential detected in response; refusing to save')
            (d/'response.json').write_bytes(raw)
            response=json.loads(raw);choice=response['choices'][0]
            content=choice['message']['content'].encode('utf-8')
            (d/'content.txt').write_bytes(content)
            contents.append(content)
            row=dict(run=i,request_sha256=digest(body),content_sha256=digest(content),same_as_first=content==contents[0],finish_reason=choice.get('finish_reason'),response_model=response.get('model'),system_fingerprint=response.get('system_fingerprint'))
            try:
                scene=json.loads(content)
                if not isinstance(scene.get('objects'),list):raise ValueError('Missing objects list')
                save(d/'scene.json',scene)
                fig,ax=plt.subplots(figsize=(7,7));plot(ax,scene,f'bedroom_1 baseline — replay {i}')
                fig.tight_layout();fig.savefig(d/'scene.png',dpi=160);plt.close(fig)
                scenes.append((i,scene));row['plot_status']='complete'
            except (ValueError,KeyError,TypeError) as e:row['plot_status']='failed: '+str(e)
            if i>1:
                diff=difflib.unified_diff(contents[0].decode().splitlines(keepends=True),content.decode().splitlines(keepends=True),fromfile='replay_01',tofile=f'replay_{i:02d}')
                (d/'difference_from_first.diff').write_text(''.join(diff))
            report['runs'].append(row);report['completed_replays']=len(contents)
            report['identical_requests']=all((out/f'replay_{j:02d}/request.json').read_bytes()==body for j in range(1,i+1))
            report['all_outputs_identical']=all(c==contents[0] for c in contents)
            report['runs_differing_from_first']=[j+1 for j,c in enumerate(contents) if c!=contents[0]]
            report['unique_outputs']=len(set(contents))
            save(out/'report.json',report)
            print(f'Completed {i}; distinct outputs: {len(set(contents))}',flush=True)
        report['status']='complete'
    except Exception as e:
        message=str(e)
        if 'token' in locals():message=message.replace(token,'[REDACTED]')
        report['status']='blocked_or_failed';report['error']=message
    finally:
        if scenes:
            fig,axes=plt.subplots(math.ceil(len(scenes)/2),2,figsize=(14,7*math.ceil(len(scenes)/2)),squeeze=False)
            for ax,(i,scene) in zip(axes.flat,scenes):plot(ax,scene,f'Replay {i}')
            for ax in list(axes.flat)[len(scenes):]:ax.axis('off')
            fig.tight_layout();fig.savefig(out/'all_scenes.png',dpi=130);plt.close(fig)
        report['difference_origin']='Identical request bodies with differing returned content; divergence is beyond client request construction. Specific provider mechanism unknown.' if len(set(contents))>1 else 'No difference observed among completed calls; this is not a guarantee of future determinism.'
        report['script_sha256']=digest(Path(__file__).read_bytes())
        save(out/'report.json',report)
        (out/'README.md').write_text('# Bedroom_1 deterministic replay\n\n'+f"Status: {report['status']}. Completed: {len(contents)}/{args.replays}. Distinct outputs: {len(set(contents))}.\n\n"+'See report.json, exact_request.json, each replay folder, difference_from_first.diff and scene.png; all_scenes.png compares the layouts. No upstream stages or adjective conditions are regenerated.\n\nDeepInfra seed documentation explicitly states: “Determinism is not guaranteed.”\nhttps://docs.deepinfra.com/api-reference/chat-completions/openai-chat-completions\n')
        print('Output:',out,flush=True)
        print(json.dumps({k:report.get(k) for k in ('status','completed_replays','identical_requests','all_outputs_identical','unique_outputs','error')},indent=2),flush=True)


if __name__=='__main__':main()
