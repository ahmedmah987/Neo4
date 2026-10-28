"""Check scripts/ple_lazy.py on a tiny random qwen4_exp (Phase 68): the model is saved in the Kaggle layout
(n-gram table pieces in files of their own), loaded the normal way and with the table memory-mapped, and
the logits and a greedy generation must agree.

usage: python scripts/check_ple_lazy.py <the real model config.json>   (needs torch, transformers>=5.16)
"""

import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(__file__))
import ple_lazy  # noqa: E402

import transformers  # noqa: E402
from transformers import AutoConfig  # noqa: E402

REAL = sys.argv[1]  # the real config.json
WORK = os.environ.get("CHECK_DIR", "/tmp/check_ple_lazy")
shutil.rmtree(WORK, ignore_errors=True)

c = json.load(open(REAL))
c.pop("quantization_config", None)
t = c["text_config"]
t.update(hidden_size=64, num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"],
         num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32,
         num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
         linear_key_head_dim=16, linear_value_head_dim=16, hc_lowrank=8, ple_embed_dim=64, ngram_vocab_size_base=500,
         split_ngram_parts=4, make_ngram_vocab_size_divisible_by=8, vocab_size=300, indexer_head_dim=16,
         indexer_n_heads=2, indexer_budget=64, bos_token_id=1, eos_token_id=2, mtp_num_hidden_layers=0)
t["mtp"] = {**t["mtp"], "num_hidden_layers": 0}
v = c["vision_config"]
v.update(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, num_position_embeddings=64)
c["image_token_id"], c["video_token_id"], c["vision_start_token_id"], c["vision_end_token_id"] = 290, 291, 292, 293
os.makedirs(os.path.join(WORK, "full"))
json.dump(c, open(os.path.join(WORK, "full", "config.json"), "w"))
config = AutoConfig.from_pretrained(os.path.join(WORK, "full"))
cls = transformers.Qwen4ExpForConditionalGeneration
torch.manual_seed(0)
model = cls(config).to(torch.bfloat16).eval()
model.save_pretrained(os.path.join(WORK, "saved"))
saved = os.path.join(WORK, "saved")

# rewrite in the Kaggle layout: table pieces in files of their own
state = {}
for f in os.listdir(saved):
    if f.endswith(".safetensors"):
        state.update(load_file(os.path.join(saved, f)))
        os.remove(os.path.join(saved, f))
for f in os.listdir(saved):
    if f.endswith(".index.json"):
        os.remove(os.path.join(saved, f))
pieces = sorted(k for k in state if ple_lazy.SHARD in k)
print("table pieces:", len(pieces), pieces[0], tuple(state[pieces[0]].shape))
assert pieces, "save_pretrained did not write table pieces"
rest = {k: v for k, v in state.items() if k not in pieces}
weight_map = {}
save_file(rest, os.path.join(saved, "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
weight_map.update(dict.fromkeys(rest, "model-00001-of-00001.safetensors"))
half = len(pieces) // 2
for i, group in enumerate((pieces[:half], pieces[half:])):
    name = f"model-ple-0000{i + 1}-of-00002.safetensors"
    save_file({k: state[k] for k in group}, os.path.join(saved, name), metadata={"format": "pt"})
    weight_map.update(dict.fromkeys(group, name))
json.dump({"metadata": {}, "weight_map": weight_map}, open(os.path.join(saved, "model.safetensors.index.json"), "w"))

ids = torch.randint(3, 280, (1, 40))
with torch.no_grad():
    ref = cls.from_pretrained(saved, dtype=torch.bfloat16).eval()
    want = ref(input_ids=ids).logits.float()
    same = model(input_ids=ids).logits.float()
print("saved and reloaded agree:", torch.allclose(want, same, atol=1e-2))

table_pieces = ple_lazy.lazy_dir(saved, os.path.join(WORK, "lazy"))
print("pieces by module:", {k: len(v) for k, v in table_pieces.items()})
ple_lazy.patch_class(transformers.models.qwen4_exp.modeling_qwen4_exp.Qwen4ExpTextNGramEmbedding)
lazy = cls.from_pretrained(os.path.join(WORK, "lazy"), dtype=torch.bfloat16, device_map={"": "cpu"}).eval()
print("rows attached:", ple_lazy.attach(lazy, table_pieces))
with torch.no_grad():
    got = lazy(input_ids=ids).logits.float()
    print("max diff", (got - want).abs().max().item())
    assert torch.allclose(got, want, atol=1e-3), "lazy table disagrees"
    a = ref.generate(input_ids=ids, max_new_tokens=8, do_sample=False)
    b = lazy.generate(input_ids=ids, max_new_tokens=8, do_sample=False)
    print("generate agrees:", torch.equal(a, b), a[0, -8:].tolist())
    assert torch.equal(a, b)
print("OK")
