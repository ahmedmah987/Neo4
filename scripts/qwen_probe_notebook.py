"""Write a Kaggle notebook that checks whether a large open-weight model can serve as the goal
advisor on Kaggle's hardware (Phase 68): it reports the machine (GPU memory, RAM, disk, library
versions), loads the model mounted from Kaggle Models, shows it three public games' first frames
and asks what each game's goal is, timing the load and every answer.

usage: python scripts/qwen_probe_notebook.py <owner>/<slug> <model handle> [--gpu=cpu|rtx6000|t4|...]
(pushed on CPU, the run reports the image's libraries without spending GPU quota; the accelerator
is then chosen in the notebook's settings and the notebook run again)
writes kaggle/qwen_probe/ (push it with scripts/kaggle_submit.py push kaggle/qwen_probe).

The probe runs as a separate script, with auto-round (which loading the AutoRound model requires and
the image lacks) installed without internet from a dataset of wheels: the competition's large
accelerator runs only with internet off. The model's ~95 GiB n-gram table is not loaded but read
from its memory-mapped files (scripts/ple_lazy.py); without a GPU of 80 GiB or more the probe only
checks the files' layout, the table and the processor (a dry run).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

kernel_id, model = sys.argv[1:3]
flags = dict(a[2:].split("=", 1) for a in sys.argv[3:] if a.startswith("--"))
ACCELERATORS = {"cpu": "none", "t4": "nvidiaTeslaT4", "p100": "nvidiaTeslaP100", "rtx6000": "nvidiaRtx6000"}
accelerator = ACCELERATORS[flags.get("gpu", "rtx6000")]
out = ROOT / "kaggle" / "qwen_probe"
out.mkdir(parents=True, exist_ok=True)


def frames() -> dict:
    import logging

    import numpy as np

    from neo4.run import arcade

    logging.disable(logging.INFO)
    result = {}
    for g in ("ls20", "tu93", "vc33"):
        obs = arcade().make(g, seed=0).reset()
        grid = np.asarray(obs.frame[-1])
        result[g] = {"grid": "".join("0123456789abcdef"[v] for v in grid.ravel()),
                     "actions": [int(getattr(a, "value", a)) for a in obs.available_actions]}
    return result


PROBE = r'''
import glob, json, os, sys, time, traceback
import numpy as np

FRAMES = json.loads(%(frames)r)
PALETTE = [(255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102), (51, 51, 51), (0, 0, 0),
           (229, 58, 163), (255, 123, 204), (249, 60, 49), (30, 147, 255), (136, 216, 241), (255, 220, 0),
           (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214)]
ACTION_TEXT = {1: "ACTION1 (a key, usually up)", 2: "ACTION2 (a key, usually down)",
               3: "ACTION3 (a key, usually left)", 4: "ACTION4 (a key, usually right)",
               5: "ACTION5 (a key: interact or a special move)", 6: "ACTION6 (click a cell)",
               7: "ACTION7 (undo)"}


def image(grid_hex):
    from PIL import Image
    grid = np.array([int(c, 16) for c in grid_hex], dtype=np.int64).reshape(64, 64)
    rgb = np.asarray(PALETTE, dtype=np.uint8)[grid]
    rgb = np.repeat(np.repeat(rgb, 8, axis=0), 8, axis=1)
    return Image.fromarray(rgb)


def find_model():
    for cfg in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        d = os.path.dirname(cfg)
        if "qwen" in d.lower():
            return d
    raise SystemExit("no model directory with a config.json under /kaggle/input")


def main():
    import torch, transformers
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen4_exp import modeling_qwen4_exp as mq
    sys.path.insert(0, "/tmp/probe")
    import ple_lazy
    print("torch", torch.__version__, "| transformers", transformers.__version__, flush=True)
    path = find_model()
    print("model dir:", path, flush=True)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    full = vram >= 80  # the model without its n-gram table is ~73 GiB
    print(f"GPU {vram:.0f} GiB, RAM {ram:.0f} GiB:", "full run" if full else "dry run (layout, table, processor)", flush=True)

    # The ~95 GiB n-gram table stays in its files, memory-mapped, and is read row by row; the
    # rest of the model goes to the GPU. (Loaded the usual way, the table's pieces and their
    # concatenation would need ~190 GiB of RAM at once.)
    t0 = time.time()
    pieces = ple_lazy.lazy_dir(path, "/tmp/lazy")
    print("table pieces:", {k: len(v) for k, v in pieces.items()}, f"({time.time() - t0:.1f} s)", flush=True)
    config = AutoConfig.from_pretrained(path)
    ple_lazy.patch_class(mq.Qwen4ExpTextNGramEmbedding)
    tc = config.text_config
    expected = {}
    with torch.device("meta"):
        for layer_id in tc.ple_layer_ids:
            m = mq.Qwen4ExpTextNGramEmbedding(tc, tc.ple_embed_dim, layer_id - 1, tc.ple_layer_ids.index(layer_id))
            expected[layer_id - 1] = m.table_rows
    for prefix, p in pieces.items():
        rows = ple_lazy.MappedRows(p, prefix)
        ids = torch.randint(0, rows.rows, (1, 1000, 16))
        t1 = time.time()
        out = rows(ids)
        print(f"{prefix}: {rows.rows} rows (model expects {expected}), {out.dtype} {tuple(out.shape)}, "
              f"16,000 random rows read in {time.time() - t1:.2f} s, finite {bool(torch.isfinite(out.float()).all())}",
              flush=True)
    processor = AutoProcessor.from_pretrained(path)
    for game, f in FRAMES.items():
        messages = [{"role": "user", "content": [{"type": "image", "image": image(f["grid"])},
                                                 {"type": "text", "text": "What is the goal?"}]}]
        inputs = processor.apply_chat_template(messages, enable_thinking=False, add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors="pt")
        print(game, "prompt tokens", inputs["input_ids"].shape[1], flush=True)
    if not full:
        print("dry run done", flush=True)
        return

    t0 = time.time()
    model = mq.Qwen4ExpForConditionalGeneration.from_pretrained("/tmp/lazy", device_map={"": 0}, dtype="auto")
    model.eval()
    print("table rows attached:", ple_lazy.attach(model, pieces), flush=True)
    print(f"load seconds {time.time() - t0:.0f}", flush=True)
    print("GPU memory after load (GiB):", round(torch.cuda.memory_allocated(0) / 2**30, 1),
          "| free:", round(torch.cuda.mem_get_info(0)[0] / 2**30, 1), flush=True)
    for game, f in FRAMES.items():
        acts = ", ".join(ACTION_TEXT[a] for a in f["actions"] if a in ACTION_TEXT)
        text = ("This is the first frame of a turn-based puzzle game on a 64x64 grid of 16 colours, "
                "shown 8 times larger. There are no instructions; the player must work out the goal. "
                f"Actions available: {acts}. What is the goal of this level, which object does the player "
                "control (if any), and which objects should it reach or click first? Answer in at most "
                "80 words, then a JSON line {\"goal\": ..., \"player\": ..., \"targets\": [...]}.")
        messages = [{"role": "user", "content": [{"type": "image", "image": image(f["grid"])},
                                                 {"type": "text", "text": text}]}]
        kwargs = dict(add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        try:
            inputs = processor.apply_chat_template(messages, enable_thinking=False, **kwargs)
        except TypeError:
            inputs = processor.apply_chat_template(messages, **kwargs)
        inputs = inputs.to("cuda:0")
        t1 = time.time()
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
        answer = processor.decode(ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n=== {game}: {time.time() - t1:.1f} s, {ids.shape[1] - inputs['input_ids'].shape[1]} tokens\n{answer}", flush=True)


try:
    main()
except SystemExit:
    raise
except Exception:
    traceback.print_exc()
    sys.exit(4)
'''

env_cell = r'''import subprocess, sys, os, shutil


def sh(cmd):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return (r.stdout + r.stderr).strip() or "(nothing)"
    except Exception as e:
        return repr(e)


print(sh("nvidia-smi || echo no GPU"))
print(sh("free -g"))
print(sh("df -h /kaggle/working /kaggle/input"))
print("cpus:", sh("nproc"))
for pkg in ("torch", "torchvision", "triton", "transformers", "accelerate", "auto_round", "vllm", "optimum", "gptqmodel"):
    try:
        m = __import__(pkg)
        print(pkg, getattr(m, "__version__", "?"))
    except Exception as e:
        print(pkg, "missing:", type(e).__name__)
print(sh("du -sh /kaggle/input/*/* 2>/dev/null | head"))
'''

probe_cell = '''PROBE = %r
LAZY = %r
import glob, os, subprocess, sys
os.makedirs("/tmp/probe", exist_ok=True)
open("/tmp/probe/probe.py", "w").write(PROBE)
open("/tmp/probe/ple_lazy.py", "w").write(LAZY)
# The competition's large accelerator runs without internet: auto-round, which loading the
# AutoRound model requires, comes from a dataset of wheels (torch and transformers as the image has them).
wheels = sorted(w for w in glob.glob("/kaggle/input/**/*.whl", recursive=True)
                if os.path.basename(w).split("-")[0] in ("auto_round", "py_cpuinfo", "threadpoolctl"))
print("wheels:", wheels)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", "/tmp/pkgs", *wheels])
env = dict(os.environ, PYTHONPATH="/tmp/pkgs")
print(subprocess.run([sys.executable, "-c", "import auto_round; print('auto_round', auto_round.__version__)"],
                     env=env, capture_output=True, text=True).stdout or "auto_round not importable")
r = subprocess.run([sys.executable, "/tmp/probe/probe.py"], env=env)
print("exit code", r.returncode)
''' % (PROBE % {"frames": json.dumps(frames())}, (ROOT / "scripts" / "ple_lazy.py").read_text())


def cell(source: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


notebook = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": accelerator, "isInternetEnabled": False, "isGpuEnabled": accelerator != "none",
                   "language": "python", "sourceType": "notebook"},
    },
    "nbformat_minor": 4,
    "nbformat": 4,
    "cells": [
        cell("# Neo4 — can a large open model advise on goals here? (feasibility probe)\n\n"
             "Reports the machine, loads the model attached under *Input → Models*, shows it three "
             "public ARC-AGI-3 games' first frames and times its answers. From "
             "https://github.com/ahmedmah987/neo4 (`scripts/qwen_probe_notebook.py`).", "markdown"),
        cell(env_cell),
        cell(probe_cell),
    ],
}
(out / "probe.ipynb").write_text(json.dumps(notebook, indent=1))
meta = {
    "id": kernel_id, "title": kernel_id.split("/")[1].replace("-", " ").title(), "code_file": "probe.ipynb",
    "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": accelerator != "none",
    "enable_tpu": False, "enable_internet": False, "keywords": [],
    "dataset_sources": ["ahmedabdalatef/neo4-autoround-wheels"], "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"], "model_sources": [model],
}
if accelerator != "none":
    meta["machine_shape"] = accelerator
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", out / "probe.ipynb", (out / "probe.ipynb").stat().st_size, "bytes")
