"""A vision-language model as an advisor on goals (Phase 58).

The model-guided agent learns what its actions do from its own transitions, but it learns what
a level asks only by completing it, and on a new game that costs ten to fifty times the human
action count. A person looking at the screen guesses the goal at once: reach the door, click
the odd one out, make the board match the picture. The advisor puts that guess in: at a level's
start, and again when the level drags on, an open-weight model is shown the frame and the list
of objects the agent sees, with what the agent has learned so far, and asked which object the
player controls and which objects to reach or click, in order.

The answer only orders the agent's exploration: advised targets are planned to or clicked
before the other untested actions, after anything the agent's own evidence says completed a
level before. It never makes the agent skip an action it would otherwise test. Without a model
(no GPU, no weights, an error, a time budget spent) the agent plays exactly as before.

The model runs in-process (``transformers``), shared by every game's thread behind a lock.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# The ARC-AGI-3 palette (arc_agi/rendering.py) and plain names for the prompt.
PALETTE = [
    (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
    (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
    (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
    (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
]
COLOUR_NAMES = [
    "white", "light grey", "grey", "dark grey", "charcoal", "black", "magenta", "pink",
    "red", "blue", "light blue", "yellow", "orange", "maroon", "green", "purple",
]
SCALE = 8  # pixels per cell in the rendered image (64 cells -> 512 pixels)
MAX_OBJECTS = 40  # objects listed in a prompt
MAX_TARGETS = 6  # targets taken from an answer
MAX_NEW_TOKENS = 160


@dataclass(slots=True)
class Advice:
    """An answer: the player's object and the targets, as indices into the listed objects."""

    goal: str = ""
    player: int | None = None
    targets: list[int] = field(default_factory=list)
    raw: str = ""
    seconds: float = 0.0


def render(grid: np.ndarray, scale: int = SCALE) -> np.ndarray:
    """The frame as an RGB image, each cell a ``scale``-pixel square, with faint cell lines
    every 8 cells so that positions can be read off."""
    pal = np.asarray(PALETTE, dtype=np.uint8)
    img = pal[np.clip(grid.astype(np.int64), 0, 15)]
    img = np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)
    for k in range(0, img.shape[0], 8 * scale):
        img[k, :, :] = (img[k, :, :].astype(np.int16) * 3 // 4).astype(np.uint8)
        img[:, k, :] = (img[:, k, :].astype(np.int16) * 3 // 4).astype(np.uint8)
    return img


def listed_objects(objs: list) -> list:
    """The objects shown to the model: not scenery, rarest kinds first, then the largest."""
    counts: dict[tuple, int] = {}
    for o in objs:
        counts[o.sig] = counts.get(o.sig, 0) + 1
    shown = [o for o in objs if not o.scenery] or list(objs)
    shown.sort(key=lambda o: (counts[o.sig], -o.area, o.y1, o.x1))
    return shown[:MAX_OBJECTS]


def describe_objects(shown: list, objs: list) -> list[str]:
    counts: dict[tuple, int] = {}
    for o in objs:
        counts[o.sig] = counts.get(o.sig, 0) + 1
    lines = []
    for i, o in enumerate(shown, 1):
        same = counts[o.sig] - 1
        lines.append(
            f"#{i}: {COLOUR_NAMES[o.color]}, {o.w}x{o.h} cells at x={o.x1}..{o.x2}, y={o.y1}..{o.y2}"
            + (f", {same} other object(s) of the same shape and colour" if same else "")
        )
    return lines


SYSTEM = (
    "You advise an agent that plays an unknown turn-based puzzle game on a 64x64 grid of coloured "
    "cells (an ARC-AGI-3 game). Nobody tells the agent the rules or the goal; a level ends when "
    "its goal is met, and every action counts against the agent, so the goal should be guessed "
    "from the picture, as a person would. Typical goals: move the player to an exit or a marked "
    "cell, make a pattern match a reference shown elsewhere on the screen, put pieces on their "
    "marks, clear or collect all objects of a kind, press the right buttons or switches. "
    "Answer with one JSON object and nothing else."
)


def build_prompt(lines: list[str], actions: list[int], notes: list[str], level: int) -> str:
    acts = []
    for a in actions:
        if a in (1, 2, 3, 4):
            acts.append(f"ACTION{a} (a key; usually {['up', 'down', 'left', 'right'][a - 1]})")
        elif a == 5:
            acts.append("ACTION5 (a key: an interaction or a special move)")
        elif a == 6:
            acts.append("ACTION6 (click a cell)")
        elif a == 7:
            acts.append("ACTION7 (undo)")
    parts = [
        f"This is level {level + 1}. The image shows the screen; x grows to the right and y "
        "downwards, from 0 to 63.",
        "Actions available: " + ", ".join(acts) + ".",
        "Objects on the screen (single-colour connected shapes; large background areas left out):",
        *lines,
    ]
    if notes:
        parts += ["What the agent has found so far:", *notes]
    parts.append(
        'Reply as JSON: {"goal": "<one sentence>", "player": <object number the keys move, or '
        'null>, "targets": [<object numbers to reach or to click, most important first, at most '
        f"{MAX_TARGETS}>]}}."
    )
    return "\n".join(parts)


def parse(text: str, n_objects: int) -> Advice | None:
    """The advice in a model's answer, or None. Object numbers out of range are dropped."""
    m = re.search(r"\{.*\}", text, re.S)
    if m is None:
        return None
    blob = m.group(0)
    data: Any = None
    for cand in (blob, blob[: blob.rfind("}") + 1]):
        try:
            data = json.loads(cand)
            break
        except ValueError:
            continue
    if not isinstance(data, dict):
        return None

    def num(v: Any) -> int | None:
        if isinstance(v, str):
            v = v.strip().lstrip("#")
            v = int(v) if v.isdigit() else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 1 <= int(v) <= n_objects:
            return int(v)
        return None

    targets: list[int] = []
    raw_targets = data.get("targets") or []
    if not isinstance(raw_targets, list):
        raw_targets = [raw_targets]
    for v in raw_targets:
        t = num(v)
        if t is not None and t not in targets:
            targets.append(t)
    return Advice(goal=str(data.get("goal", ""))[:200], player=num(data.get("player")),
                  targets=targets[:MAX_TARGETS], raw=text[:600])


class Advisor:
    """No model: every question goes unanswered."""

    name = "none"

    def __init__(self) -> None:
        self.stats: dict[str, Any] = {"queries": 0, "answers": 0, "failures": 0, "seconds": 0.0}

    @property
    def available(self) -> bool:
        return False

    def advise(self, grid: np.ndarray, prompt: str, n_objects: int) -> Advice | None:
        return None


class ScriptedAdvisor(Advisor):
    """Answers from a function of the prompt (tests and offline experiments)."""

    name = "scripted"

    def __init__(self, answer) -> None:
        super().__init__()
        self.answer = answer
        self.prompts: list[str] = []

    @property
    def available(self) -> bool:
        return True

    def advise(self, grid: np.ndarray, prompt: str, n_objects: int) -> Advice | None:
        self.stats["queries"] += 1
        self.prompts.append(prompt)
        advice = parse(self.answer(prompt), n_objects)
        if advice is not None:
            self.stats["answers"] += 1
        return advice


class ModelAdvisor(Advisor):
    """An image-text model from a local directory, loaded on first use, one query at a time.

    ``time_budget`` caps the seconds spent generating, over all games: past it the advisor
    stops answering and the agents play on without it. Repeated failures switch it off too.
    """

    name = "model"

    def __init__(self, path: str, time_budget: float = 4 * 3600.0, use_image: bool = True) -> None:
        super().__init__()
        self.path = path
        self.time_budget = time_budget
        self.use_image = use_image
        self.lock = threading.Lock()
        self.model: Any = None
        self.processor: Any = None
        self.broken = False
        self.load_seconds = 0.0

    @property
    def available(self) -> bool:
        return not self.broken and self.stats["seconds"] < self.time_budget

    def _load(self) -> None:
        if self.model is not None or self.broken:
            return
        t0 = time.monotonic()
        try:
            import torch
            import transformers

            path = resolve(self.path)
            if path is None:
                raise FileNotFoundError(f"no model directory for {self.path!r}")
            self.stats["model_path"] = path
            self.stats["dtype"] = os.environ.get("NEO4_ADVISOR_DTYPE", "") or "auto"
            # bfloat16 where the GPU has it natively, else float16; NEO4_ADVISOR_DTYPE overrides (Gemma 3
            # overflows in float16, so on a T4 a small Gemma runs in float32)
            native_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
            dtype = getattr(torch, os.environ.get("NEO4_ADVISOR_DTYPE", "") or
                            ("bfloat16" if native_bf16 else "float16"))
            # Spread the weights over every GPU, each filled to at most 70%, so that the activations of
            # a long prompt still fit (device_map="auto" alone filled the first T4 and ran out of memory)
            max_memory = None
            if torch.cuda.is_available():
                max_memory = {i: int(torch.cuda.get_device_properties(i).total_memory * 0.70)
                              for i in range(torch.cuda.device_count())}
                max_memory["cpu"] = 64 * 2**30
            errors = []
            # image-text models first; a text-only model is shown the object list without the image
            for cls_name, image in (("AutoModelForImageTextToText", True), ("AutoModelForMultimodalLM", True),
                                    ("AutoModelForVision2Seq", True), ("AutoModelForCausalLM", False)):
                cls = getattr(transformers, cls_name, None)
                if cls is None:
                    continue
                try:
                    self.model = cls.from_pretrained(path, torch_dtype=dtype, device_map="auto",
                                                     max_memory=max_memory)
                    self.use_image = self.use_image and image
                    self.stats["model_class"] = cls_name
                    break
                except Exception as e:
                    errors.append(f"{cls_name}: {e!r}"[:200])
            if self.model is None:
                raise RuntimeError("; ".join(errors))
            self.processor = transformers.AutoProcessor.from_pretrained(path)
            self.model.eval()
        except Exception as e:  # no GPU, no weights, an incompatible library: play without
            self.broken = True
            self.model = None
            self.stats["load_error"] = repr(e)[:600]
        self.load_seconds = time.monotonic() - t0
        self.stats["load_seconds"] = round(self.load_seconds, 1)

    def advise(self, grid: np.ndarray, prompt: str, n_objects: int) -> Advice | None:
        if not self.available:
            return None
        with self.lock:
            self._load()
            if self.broken or not self.available:
                return None
            self.stats["queries"] += 1
            t0 = time.monotonic()
            try:
                text = self._generate(grid, prompt)
            except Exception as e:
                self.stats["failures"] += 1
                self.stats["last_error"] = repr(e)[:300]
                if self.stats["failures"] >= 5 and self.stats["answers"] == 0:
                    self.broken = True
                return None
            finally:
                self.stats["seconds"] += time.monotonic() - t0
        advice = parse(text, n_objects)
        if advice is None:
            self.stats["unparsed"] = self.stats.get("unparsed", 0) + 1
            self.stats["last_unparsed"] = text[:300]
            return None
        advice.seconds = time.monotonic() - t0
        self.stats["answers"] += 1
        return advice

    def _generate(self, grid: np.ndarray, prompt: str) -> str:
        import torch
        from PIL import Image

        if self.use_image:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "image", "image": Image.fromarray(render(grid))},
                                             {"type": "text", "text": prompt}]},
            ]
        else:  # a text-only model: the screen as text, plain strings
            text = prompt.replace("The image shows the screen;", "The screen is shown below;")
            messages = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": ascii_grid(grid) + "\n\n" + text}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        return self.processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def resolve(path: str, base: str = "/kaggle/input") -> str | None:
    """A model directory: ``path`` itself, or for ``auto:<name>|<name>...`` the first directory
    under ``base`` whose path contains a name (tried in order) and that holds a config.json."""
    if not path.startswith("auto:"):
        return path if os.path.isdir(path) else None
    found = []
    for root, dirs, files in os.walk(base):
        dirs.sort()
        if "config.json" in files:
            found.append(root)
    for name in path[5:].lower().split("|"):
        for root in found:
            if name and name in root.lower():
                return root
    return None


def ascii_grid(grid: np.ndarray) -> str:
    """The frame as text for a model without vision: one hexadecimal colour code per cell."""
    legend = ", ".join(f"{i:x}={n}" for i, n in enumerate(COLOUR_NAMES))
    rows = ["".join(f"{int(v) & 15:x}" for v in row) for row in grid]
    return f"The screen, one character per cell (row y=0 first; {legend}):\n" + "\n".join(rows)


_SHARED: Advisor | None = None
_SHARED_LOCK = threading.Lock()


def shared() -> Advisor:
    """The process's advisor: a model from ``NEO4_ADVISOR`` (a directory), else none."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            path = os.environ.get("NEO4_ADVISOR", "")
            budget = float(os.environ.get("NEO4_ADVISOR_BUDGET", str(2.5 * 3600)))
            _SHARED = ModelAdvisor(path, time_budget=budget) if path else Advisor()
        return _SHARED


def png(grid: np.ndarray) -> bytes:
    """The rendered frame as PNG bytes (for logs and the paper's figures)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(render(grid)).save(buf, format="PNG")
    return buf.getvalue()
