"""Single-request-at-a-time, localhost OpenAI chat endpoint for text models.

Uses greedy decoding and eager attention. JSON output is prompted, not
grammar constrained; DSPy validates and may reject malformed model output.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    parser.add_argument('--revision', required=True, help='Immutable model commit SHA')
    parser.add_argument('--port', type=int, default=7501)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-new-tokens', type=int, default=8192)
    parser.add_argument('--metadata', default='outputs/server-metadata.json')
    args = parser.parse_args()
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, device_map='auto',
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        attn_implementation='eager',
    ).eval()
    metadata = dict(model=args.model, revision=args.revision, seed=args.seed,
                    torch=torch.__version__, transformers=transformers.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                    decoding='greedy', attention='eager')
    path = Path(args.metadata)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.send_json(200, {'data': [{'id': args.model}], 'metadata': metadata})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                self.send_json(404, {'error': {'message': 'Unknown endpoint'}})
                return
            try:
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if request.get('model') != args.model:
                    raise ValueError('Requested model does not match loaded model')
                messages = request['messages']
                if request.get('response_format', {}).get('type') in ('json_object', 'json_schema'):
                    messages = [{'role': 'system', 'content': 'Respond with valid JSON only.'}] + messages
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors='pt').to(model.device)
                limit = min(int(request.get('max_tokens') or args.max_new_tokens), args.max_new_tokens)
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=False, num_beams=1,
                                            max_new_tokens=limit, use_cache=True,
                                            pad_token_id=tokenizer.eos_token_id)
                tokens = output[0, inputs.input_ids.shape[1]:]
                content = tokenizer.decode(tokens, skip_special_tokens=True)
                self.send_json(200, {
                    'id': 'local', 'object': 'chat.completion', 'created': 0, 'model': args.model,
                    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content},
                                 'finish_reason': 'length' if len(tokens) == limit else 'stop'}],
                    'usage': {'prompt_tokens': inputs.input_ids.shape[1],
                              'completion_tokens': len(tokens), 'total_tokens': output.shape[1]},
                })
            except Exception as exc:
                # Avoid logging request contents or authentication headers.
                self.send_json(400, {'error': {'message': str(exc), 'type': type(exc).__name__}})

    print(f'Ready: http://127.0.0.1:{args.port}/v1', flush=True)
    HTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
