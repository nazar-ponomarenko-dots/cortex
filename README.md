# Cortex

An FFN that builds its own weights for every token, plus adaptive depth: the block decides on its own whether to rework a token. An experiment on top of SmolLM-135M.

[Читать на русском](README_RU.md)

**The short version:** the block trains, it can be inserted into a pretrained model without any loss of quality, but it does not get better than a plain FFN. The visible 0.06 loss gain comes from noise in the weights: replace those same FFNs with their own weights plus noise and you get exactly as much. Meanwhile it has 3.7× the parameters and runs 2.2× longer (and the variant that can actually stand in for an FFN runs 3.5× longer). The two experiments are written up in [RESEARCH.md](RESEARCH.md).

**A fair warning about the numbers.** I'm not fully sure how much to trust them: the training was probably too short, and the text I trained on (a slice of an old Wikipedia dump) may not have been the best choice. At 6000 steps both curves are still creeping down, so the differences below are differences in learning speed over a short stretch, not the final quality of the architectures. Read this as a first attempt by a hobbyist, not as a solid study.

**Year:** November 2025, my early experiment with neural network architecture. Trained and measured properly in September 2026.

![Training from scratch](docs/from_scratch.png)

## What it is

A normal FFN uses the same weights for every token. Cortex builds the weights for each token from scratch. A small network looks at the token and decides:

- how much of the token to let into the block at all,
- how to mix 16 shared basis matrices into its own up-projection,
- what small weight correction to add (LoRA style), what bias and what gain per neuron to apply,
- how many times to rework the token: zero, one or two.

That last point is the adaptive depth. There is also an optional gate branch: it lets the block copy the model's SwiGLU FFN. Without it the block cannot stand in for a trained FFN, because its output is off by 9× in scale.

## What came out of it

Two experiments on the pretrained SmolLM-135M, same budget (1500 steps, 14M tokens), loss on validation data:

| What we did | At the start | At the end |
|---|---|---|
| Nothing, plain fine-tuning | 3.7161 | **3.2446** |
| **Replace:** 1/3 of FFNs → Cortex with random weights | 19.54 | 6.7282 |
| **Replace:** Cortex initialised from the FFN, no gate branch | 12.5551 | 4.9426 |
| **Replace:** Cortex initialised from the FFN, with gate branch | 3.7161 | **3.2453** |
| **Replace, control:** 1/3 of FFNs → their own weights plus noise | 3.7154 | **3.1807** |
| **Add:** a Cortex branch next to the FFN | 3.7161 | **3.1788** |
| **Add:** the same with a gate branch | 3.7161 | 3.2078 |

- A random Cortex block in place of an FFN breaks the model, and 1500 steps do not bring it back.
- With proper initialisation the replacement goes smoothly, but gives exactly what plain fine-tuning gives: 3.2453 against 3.2446.
- Adding a branch gives 3.1788 — as much as noise in the FFN weights (3.1807). The gain seems to come from perturbing the weights, not from the architecture.
- A from-scratch run over 6000 steps: Cortex (496M) 5.807 against the plain model (134.5M) 5.834 — level.
- Adaptive depth does work: 33.2% of tokens are left alone, 41.8% are reworked once, 24.9% twice.

![Replacing FFNs](docs/replacement.png)
![Adding a branch](docs/addition.png)
![Cost](docs/cost.png)

## Files

- `cortex_model.py` — the Cortex block, the gate branch, initialisation from the FFN it replaces, the replace and add routines, and the model classes.
- `train.py` — training in three modes: add a branch, replace FFNs, train from scratch.
- `chat.py` — a simple terminal chat for a trained checkpoint.
- `RESEARCH.md` / `RESEARCH_RU.md` — the full write-up: two experiments, method, results, what we did not check.
- `docs/` — charts.

## How to run

```bash
pip install -r requirements.txt

python train.py --data wiki.txt --mode add                  # a branch next to the FFN
python train.py --data wiki.txt --mode replace --warm-start # replace 1/3 of the FFNs
python train.py --data wiki.txt --mode scratch              # from scratch

python chat.py                    # the from-scratch checkpoint (meaningless text, see Status)
python chat.py smollm-cortex-add  # the add-a-branch checkpoint — readable English
```

A trained checkpoint loads with plain `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`; the right class is picked up automatically through `auto_map`.

The weights themselves are not in the repository: the checkpoints live locally (`cortex-long-fixed` ~950 MB and `cortex-parallel-ckpt` ~1 GB) and are wired in through the `smollm-cortex` and `smollm-cortex-add` symlinks.

## Status

This is research, not a usable model. The checkpoint from the from-scratch run (6000 steps) writes text that looks like English but means nothing. The weights are not in this repository — they are about 950 MB, kept locally and wired in through the `smollm-cortex` symlink. Along the way we found mistakes in the code and in the data preparation that had inflated our first published numbers. Everything was redone, so the numbers on this page are the corrected ones.

## License

MIT

---

P.S. This README was edited by a neural network — the content and meaning are the author's own.
