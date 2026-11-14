"""Check notebook reproducibility and exercises with instructor substitutions."""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from percent_to_ipynb import split_cells


def solved_source():
    source = (ROOT / 'lab_transformer.py').read_text()
    bodies = json.loads((ROOT / 'tools/reference_bodies.json').read_text())
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    edits = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in bodies:
            stub = node.body[-1]
            assert isinstance(stub, ast.Expr) and isinstance(stub.value, ast.Constant) and stub.value.value is Ellipsis
            body = ''.join('    ' + line + '\n' for line in bodies[node.name].splitlines())
            edits.append((stub.lineno - 1, stub.end_lineno, body))
    assert len(edits) == len(bodies)
    for start, stop, body in reversed(edits):
        lines[start:stop] = [body]
    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', action='store_true', help='download checkpoint and compare real inference')
    args = parser.parse_args()
    source = (ROOT / 'lab_transformer.py').read_text()
    compile(source, 'lab_transformer.py', 'exec')
    notebook = json.loads((ROOT / 'lab_transformer.ipynb').read_text())
    assert notebook['cells'] == split_cells(source), 'Regenerate notebook with make notebook'
    namespace = {'__name__': 'lab_checks'}
    exec(compile(solved_source(), 'lab_transformer_solved.py', 'exec'), namespace)
    for name in ('test_primitives', 'test_masks', 'test_attention', 'test_tiny_model'):
        namespace[name]()
    # Exercise generation stopping independently of random model outputs.
    import torch
    from types import SimpleNamespace
    original_stack, original_decode = namespace['transformer_stack'], namespace['decode_logits']
    namespace['transformer_stack'] = lambda *a, **k: None
    namespace['decode_logits'] = lambda *a: torch.tensor([[[0., 10., 0.]]])
    cfg = SimpleNamespace(decoder_start_token_id=0, eos_token_id=1)
    ids = torch.tensor([[2, 1]])
    assert namespace['greedy_generate'](ids, torch.ones_like(ids), {}, cfg).tolist() == [[0, 1]]
    assert namespace['greedy_generate'](ids, torch.ones_like(ids), {}, cfg, 0).tolist() == [[0]]
    namespace['decode_logits'] = lambda *a: torch.tensor([[[0., 0., 10.]]])
    assert namespace['greedy_generate'](ids, torch.ones_like(ids), {}, cfg, 3).tolist() == [[0, 2, 2, 2]]
    namespace['transformer_stack'], namespace['decode_logits'] = original_stack, original_decode
    print('✓ syntax, notebook, exercises, EOS and token limit')
    if args.pretrained:
        config, weights, tokenizer = namespace['load_checkpoint']()
        namespace['compare_pretrained'](config, weights, tokenizer)


if __name__ == '__main__':
    main()
