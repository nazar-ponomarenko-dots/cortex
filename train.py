import argparse
import os
import shutil
import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
from cortex_model import CortexForCausalLM, CortexParallelForCausalLM, replace_ffn_with_cortex


def pack_ids(ids, seq_len):
    num = (len(ids) - 1) // (seq_len + 1)
    t = torch.tensor(ids[:num * (seq_len + 1)], dtype=torch.long).view(num, seq_len + 1)
    return t[:, :-1], t[:, 1:]


def tokenize_file(path, tokenizer, max_chars, chunk_chars=2_000_000):
    ids = []
    taken = 0
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        while taken < max_chars:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            ids.extend(tokenizer(chunk)['input_ids'])
            taken += len(chunk)
            print(f'tokenized {taken} chars', flush=True)
    return ids


@torch.no_grad()
def eval_loss(model, val_dl, device):
    model.eval()
    total, count = 0.0, 0
    for x, y in val_dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        total += out.loss.item()
        count += 1
    model.train()
    return total / count


def _save(model, tok, cortex_config, out):
    name = model.__class__.__name__
    model.config.cortex_config = cortex_config
    model.config.architectures = [name]
    model.config.auto_map = {'AutoModelForCausalLM': f'modeling_cortex.{name}'}
    model.save_pretrained(out)
    tok.save_pretrained(out)
    shutil.copyfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cortex_model.py'),
                    os.path.join(out, 'modeling_cortex.py'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='path to a plain-text training file')
    p.add_argument('--base-model', default='HuggingFaceTB/SmolLM-135M')
    p.add_argument('--mode', default='add', choices=['add', 'replace', 'scratch'],
                   help='"add" trains a Cortex branch next to the pretrained FFN, '
                        '"replace" swaps every n-th FFN for Cortex, "scratch" trains randomly initialised Cortex')
    p.add_argument('--warm-start', action='store_true',
                   help='with --mode replace: initialise Cortex from the FFN it replaces')
    p.add_argument('--every-n-layers', type=int, default=3)
    p.add_argument('--steps', type=int, default=1500)
    p.add_argument('--seq-len', type=int, default=128)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--grad-accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--out', default='./smollm-cortex')
    p.add_argument('--max-chars', type=int, default=60_000_000)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)

    tok = AutoTokenizer.from_pretrained(args.base_model)
    print(f'tokenizing {args.data}...')
    ids = tokenize_file(args.data, tok, args.max_chars)
    print(f'{len(ids)} tokens')
    if len(ids) < 8 * args.seq_len:
        raise SystemExit('not enough text to train on')

    split = int(len(ids) * 0.95)
    xtr, ytr = pack_ids(ids[:split], args.seq_len)
    xva, yva = pack_ids(ids[split:split + 200_000], args.seq_len)
    train_dl = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(TensorDataset(xva[:512], yva[:512]), batch_size=args.batch_size)

    config = AutoConfig.from_pretrained(args.base_model)
    config.cortex_config = {
        'num_bases': 16, 'lora_rank': 4, 'max_iterations': 2,
        'every_n_layers': args.every_n_layers,
    }
    if args.mode == 'scratch':
        model = CortexForCausalLM(config).to(torch.bfloat16)
    elif args.warm_start:
        model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.bfloat16)
        replace_ffn_with_cortex(model, every_n_layers=args.every_n_layers, num_bases=16,
                                lora_rank=4, max_iterations=2, warm_start=True)
        model.__class__ = CortexForCausalLM
    else:
        cls = CortexParallelForCausalLM if args.mode == 'add' else CortexForCausalLM
        model = cls.from_pretrained(args.base_model, config=config, dtype=torch.bfloat16)

    print(f'parameters: {sum(q.numel() for q in model.parameters()) / 1e6:.1f}M')
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()

    opt = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=0.01)
    opt.zero_grad()
    history = []
    it = iter(train_dl)
    start = time.time()
    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(train_dl)
            x, y = next(it)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model(input_ids=x, labels=y)
        (out.loss / args.grad_accum).backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        if step % args.eval_every == 0 or step == 1:
            vl = eval_loss(model, val_dl, device)
            history.append({'step': step, 'train_loss': out.loss.item(), 'val_loss': vl})
            print(f'step {step} train {out.loss.item():.4f} val {vl:.4f} elapsed {time.time() - start:.0f}s', flush=True)

    print(f'final val loss: {history[-1]["val_loss"]:.4f}')
    _save(model, tok, config.cortex_config, args.out)
    print(f'model saved to {args.out}')


if __name__ == '__main__':
    main()
