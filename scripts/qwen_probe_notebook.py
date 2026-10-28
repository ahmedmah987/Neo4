"""Write a Kaggle notebook that checks whether a large open-weight model can serve as the goal
advisor on Kaggle's hardware (Phase 68): it reports the machine (GPU memory, RAM, disk, library
versions), loads the model mounted from Kaggle Models, shows it three public games' first frames
and asks what each game's goal is, timing the load and every answer.

usage: python scripts/qwen_probe_notebook.py <owner>/<slug> <model handle> [--gpu=cpu|rtx6000|t4|...]
(pushed on CPU, the run reports the image's libraries without spending GPU quota; the accelerator
is then chosen in the notebook's settings and the notebook run again)
writes kaggle/qwen_probe/ (push it with scripts/kaggle_submit.py push kaggle/qwen_probe).

The probe runs as a separate script, first with the image's libraries and, if loading fails, once
more after installing auto-round alone, without its dependencies (the probe notebook has internet;
a submission does not, so a second pass would mean shipping that wheel).
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
    print("torch", torch.__version__, "| transformers", transformers.__version__, flush=True)
    path = find_model()
    print("model dir:", path, flush=True)
    from transformers import AutoProcessor
    import transformers as tf
    vram = int(torch.cuda.get_device_properties(0).total_memory / 2**30) if torch.cuda.is_available() else 0
    ram = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30)
    max_memory = {i: f"{int(torch.cuda.get_device_properties(i).total_memory / 2**30 * 0.92)}GiB"
                  for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = f"{int(ram * 0.85)}GiB"
    print("max_memory", max_memory, flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(path)
    model, err = None, []
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText", "AutoModelForCausalLM"):
        cls = getattr(tf, name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(path, device_map="auto", max_memory=max_memory, torch_dtype="auto")
            print("loaded with", name, flush=True)
            break
        except Exception as e:
            err.append(f"{name}: {e!r}"[:800])
    if model is None:
        print("LOAD FAILED:\n" + "\n".join(err), flush=True)
        sys.exit(3)
    print(f"load seconds {time.time() - t0:.0f}", flush=True)
    if torch.cuda.is_available():
        print("GPU memory after load (GiB):", [round(torch.cuda.memory_allocated(i) / 2**30, 1)
                                               for i in range(torch.cuda.device_count())], flush=True)
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
        inputs = inputs.to(model.device)
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
for pkg in ("torch", "transformers", "accelerate", "auto_round", "vllm", "optimum", "gptqmodel"):
    try:
        m = __import__(pkg)
        print(pkg, getattr(m, "__version__", "?"))
    except Exception as e:
        print(pkg, "missing:", type(e).__name__)
print(sh("du -sh /kaggle/input/*/* 2>/dev/null | head"))
'''

probe_cell = '''PROBE = %r
open("/kaggle/working/probe.py", "w").write(PROBE)
import os, subprocess, sys
r = subprocess.run([sys.executable, "/kaggle/working/probe.py"])
print("first pass exit code", r.returncode)
if r.returncode != 0:
    print("installing auto-round alone (torch and transformers as the image has them), then trying once more")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", "/tmp/pkgs",
                    "auto-round", "py-cpuinfo", "threadpoolctl"])
    env = dict(os.environ, PYTHONPATH="/tmp/pkgs")
    r = subprocess.run([sys.executable, "/kaggle/working/probe.py"], env=env)
    print("second pass exit code", r.returncode)
''' % (PROBE % {"frames": json.dumps(frames())})


def cell(source: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "metadata": {}, "source": source}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


notebook = {
    "metadata": {
        "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": accelerator, "isInternetEnabled": True, "isGpuEnabled": accelerator != "none",
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
    "enable_tpu": False, "enable_internet": True, "keywords": [], "dataset_sources": [], "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"], "model_sources": [model],
}
if accelerator != "none":
    meta["machine_shape"] = accelerator
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print("wrote", out / "probe.ipynb", (out / "probe.ipynb").stat().st_size, "bytes")
