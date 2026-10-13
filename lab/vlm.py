"""Bước 3–7: nạp Qwen3-VL-4B (QLoRA), dựng ví dụ + mask loss, setup LoRA/đóng băng, train, suy luận."""
import functools
import json
import random
import re
import time
from pathlib import Path

import torch
from PIL import Image

from lab.labels import build_prompt, objects_to_answer
from scripts.labcommon import read_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
# Regex khớp TOÀN BỘ tên module (peft dùng re.fullmatch). Neo vào language_model.layers / visual.blocks
# để không dính linear_fc1/fc2 của projector.
LLM_TARGETS = r".*language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
VIT_TARGETS = r".*visual\.blocks\.\d+\.(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])"
# should_convert_module() của transformers 5.x dùng re.match (neo ở đầu tên): "visual" KHÔNG khớp
# "model.visual.blocks.0.attn.qkv" nên ViT vẫn bị lượng tử hoá. Phải ghi đủ tiền tố "model.visual".
SKIP_QUANT = ["model.visual", "lm_head"]


# ---------------------------------------------------------------- nạp model

def load_processor(model_id=MODEL_ID, padding_side="left"):
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    proc.tokenizer.padding_side = padding_side
    return proc


def load_model(model_id=MODEL_ID, quantize=None):
    """quantize=None: 4-bit khi có CUDA (bitsandbytes cần GPU), ngược lại nạp thường để chạy thử."""
    from transformers import BitsAndBytesConfig, Qwen3VLForConditionalGeneration
    if quantize is None:
        quantize = torch.cuda.is_available()
    kw = {}
    if quantize:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True, llm_int8_skip_modules=SKIP_QUANT)
    cuda = torch.cuda.is_available()
    # T4: fp16. Không có CUDA (chỉ để chạy thử model tí hon trên CPU): fp32.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype=torch.float16 if cuda else torch.float32,
        device_map={"": 0} if cuda else None, **kw)
    if quantize:
        check_quantization(model)
    return model


def check_quantization(model):
    """ViT (kể cả projector) và lm_head phải còn fp16; LLM phải là 4-bit."""
    import bitsandbytes as bnb
    q = [n for n, m in model.named_modules() if isinstance(m, bnb.nn.Linear4bit)]
    bad = [n for n in q if ".visual." in f".{n}" or n.endswith("lm_head")]
    assert not bad, f"ViT/lm_head bị lượng tử hoá: {bad[:3]}"
    assert any("language_model" in n for n in q), "LLM không được lượng tử hoá"
    return len(q)


# ---------------------------------------------------------------- dữ liệu

def user_message(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


def encode_example(processor, image, prompt, answer):
    """Chuỗi đầy đủ (user + assistant) và labels chỉ trên phần trả lời."""
    user = user_message(image, prompt)
    prompt_enc = processor.apply_chat_template(
        [user], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    full_enc = processor.apply_chat_template(
        [user, {"role": "assistant", "content": [{"type": "text", "text": answer}]}],
        add_generation_prompt=False, tokenize=True, return_dict=True, return_tensors="pt")
    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ")
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = -100
    out = dict(full_enc)
    out["labels"] = labels
    return out


PAD_VALUES = {"input_ids": None, "attention_mask": 0, "mm_token_type_ids": 0, "labels": -100}


def collate(batch, pad_token_id):
    """Đệm phải các chuỗi token; pixel_values / image_grid_thw nối theo chiều 0 (không pad)."""
    width = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for key, val in PAD_VALUES.items():
        if key not in batch[0]:
            continue
        val = pad_token_id if val is None else val
        rows = []
        for b in batch:
            t = b[key][0]
            rows.append(torch.cat([t, t.new_full((width - t.shape[0],), val)]))
        out[key] = torch.stack(rows)
    out["pixel_values"] = torch.cat([b["pixel_values"] for b in batch], 0)
    out["image_grid_thw"] = torch.cat([b["image_grid_thw"] for b in batch], 0)
    return out


def subsample(entries, fraction, seed=42):
    if fraction >= 1.0:
        return list(entries)
    rng = random.Random(seed)
    k = max(1, round(len(entries) * fraction))
    return sorted(rng.sample(list(entries), k), key=lambda e: e["image_id"])


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, entries, data_dir, processor, coord="norm1000"):
        self.entries, self.data_dir, self.processor, self.coord = entries, Path(data_dir), processor, coord

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        image = Image.open(self.data_dir / e["image_path"]).convert("RGB")
        prompt = build_prompt(self.coord, e["width"], e["height"])
        answer = objects_to_answer(e["objects"], self.coord, e["width"], e["height"])
        return encode_example(self.processor, image, prompt, answer)


# ---------------------------------------------------------------- LoRA / đóng băng

def lora_regex(targets):
    parts = {"llm": [LLM_TARGETS], "llm+vit": [LLM_TARGETS, VIT_TARGETS]}[targets]
    return "|".join(f"(?:{p})" for p in parts)


def matched_modules(model, targets):
    rx = re.compile(lora_regex(targets))
    return [n for n, _ in model.named_modules() if rx.fullmatch(n)]


def is_projector(name):
    return any(k in name for k in PROJECTOR_KEYS)


def apply_setup(model, lora, train_projector):
    """lora: {r, alpha, dropout, targets: llm | llm+vit}. Trả (peft_model, info)."""
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad = False
    model.enable_input_require_grads()            # cần cho gradient checkpointing
    matched = matched_modules(model, lora["targets"])
    cfg = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                     target_modules=lora_regex(lora["targets"]), bias="none")
    pm = get_peft_model(model, cfg)
    if train_projector:
        for n, p in pm.named_parameters():
            if is_projector(n):
                p.data = p.data.float()
                p.requires_grad = True
    for p in pm.parameters():                     # GradScaler không unscale được gradient fp16
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    info = {
        "matched_modules": len(matched),
        "trainable_params": sum(p.numel() for p in pm.parameters() if p.requires_grad),
        "lora_params": sum(p.numel() for n, p in pm.named_parameters() if p.requires_grad and "lora_" in n),
        "projector_params": sum(p.numel() for n, p in pm.named_parameters()
                                if p.requires_grad and is_projector(n)),
    }
    return pm, info


def extra_state(model):
    """Tham số huấn luyện đầy đủ không nằm trong adapter peft (projector ở E2/E3)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and "lora_" not in n}


def save_checkpoint(model, out_dir):
    from safetensors.torch import save_file
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir)               # chỉ adapter LoRA
    extra = extra_state(model)
    if extra:
        save_file(extra, str(out_dir / "extra_trainable.safetensors"))
    return len(extra)


def load_extra(model, adapter_dir):
    """Nạp lại projector đã huấn luyện. Trả số tensor đã nạp (0 nếu không có)."""
    from safetensors.torch import load_file
    path = Path(adapter_dir) / "extra_trainable.safetensors"
    if not path.exists():
        return 0
    sd = load_file(str(path))
    params = dict(model.named_parameters())
    missing = [k for k in sd if k not in params]
    if missing:
        raise KeyError(f"extra_trainable có tên không khớp model: {missing[:3]}")
    with torch.no_grad():
        for k, v in sd.items():
            params[k].copy_(v.to(params[k].dtype))
    return len(sd)


# ---------------------------------------------------------------- huấn luyện

def resume_checkpoint(model, out_dir):
    """Nạp lại adapter + projector đã lưu giữa chừng. Trả progress {step, minutes, peak_vram_gb} hoặc None."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    out_dir = Path(out_dir)
    prog = out_dir / "progress.json"
    if not prog.exists():
        return None
    set_peft_model_state_dict(model, load_file(str(out_dir / "adapter_model.safetensors")))
    load_extra(model, out_dir)
    return json.loads(prog.read_text(encoding="utf-8"))


def train(model, processor, entries, cfg, data_dir, out_dir, coord="norm1000", log=print,
          num_workers=2, save_every=100, progress=None):
    """Vòng train theo GUIDE bước 6. cfg: lr, batch_size, grad_accum, max_steps, seed.

    progress (từ resume_checkpoint): chạy tiếp từ bước đã lưu. AdamW khởi tạo lại moment
    (không lưu trạng thái optimizer để checkpoint trên Drive nhỏ), lịch lr được tua tới đúng bước.
    """
    progress = progress or {"step": 0, "minutes": 0.0, "peak_vram_gb": 0.0}
    from transformers import get_cosine_schedule_with_warmup
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    device = next(p for p in model.parameters() if p.device.type != "meta").device
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    steps, accum = cfg["max_steps"], cfg["grad_accum"]
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(0.05 * steps)), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)

    ds = SFTDataset(entries, data_dir, processor, coord)
    gen = torch.Generator().manual_seed(cfg["seed"] + progress["step"])
    pad_id = processor.tokenizer.pad_token_id

    def batches():
        while True:
            loader = torch.utils.data.DataLoader(
                ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen, drop_last=True,
                collate_fn=functools.partial(collate, pad_token_id=pad_id), num_workers=num_workers,
                persistent_workers=False)
            yield from loader

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for _ in range(progress["step"]):
        sched.step()
    if progress["step"]:
        log(f"chạy tiếp từ bước {progress['step']} ({progress['minutes']:.1f} phút đã train)")
    it, t0 = batches(), time.time() - progress["minutes"] * 60
    logf = open(out_dir / "train_log.jsonl", "a" if progress["step"] else "w")
    for step in range(progress["step"] + 1, steps + 1):
        total = 0.0
        for _ in range(accum):
            batch = {k: v.to(device) for k, v in next(it).items()}
            with torch.autocast(device.type, dtype=torch.float16):
                loss = model(**batch).loss / accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn ở bước {step}: {loss.item()}")
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        rec = {"step": step, "loss": total, "lr": sched.get_last_lr()[0], "grad_norm": gnorm,
               "minutes": (time.time() - t0) / 60}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if step % 10 == 0 or step == 1:
            log(f"step {step:4d}/{steps} loss={total:.4f} lr={rec['lr']:.2e} "
                f"gnorm={gnorm:.2f} {rec['minutes']:.1f} phút")
        if step % save_every == 0 or step == steps:
            save_checkpoint(model, out_dir)
            (out_dir / "progress.json").write_text(json.dumps(
                {"step": step, "minutes": rec["minutes"], "peak_vram_gb": _peak(progress)}), encoding="utf-8")
    logf.close()
    run = {
        "trainable_params": sum(p.numel() for p in params),
        "peak_vram_gb": _peak(progress),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "n_train_images": len(entries), "coord": coord, "cfg": cfg,
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    return run


def _peak(progress):
    cur = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    return round(max(cur, progress["peak_vram_gb"]), 2)


# ---------------------------------------------------------------- suy luận

def load_for_inference(adapter_dir=None, model_id=MODEL_ID, quantize=None):
    model = load_model(model_id, quantize=quantize)
    n_extra = 0
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir))
        n_extra = load_extra(model, adapter_dir)
    model.eval()
    return model, n_extra


@torch.no_grad()
def predict(model, processor, entries, data_dir, out_path, coord="norm1000", batch_size=4,
            max_new_tokens=256, log=print):
    """Ghi {"image_id", "raw"} từng dòng; chạy tiếp được nếu bị ngắt giữa chừng."""
    out_path, data_dir = Path(out_path), Path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["image_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    todo = [e for e in entries if e["image_id"] not in done]
    processor.tokenizer.padding_side = "left"
    device = next(p for p in model.parameters() if p.device.type != "meta").device
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            chunk = todo[i:i + batch_size]
            msgs = [[user_message(Image.open(data_dir / e["image_path"]).convert("RGB"),
                                  build_prompt(coord, e["width"], e["height"]))] for e in chunk]
            inputs = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", processor_kwargs={"padding": True}).to(device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for e, raw in zip(chunk, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            n = i + len(chunk)
            if n % (batch_size * 10) == 0 or n == len(todo):
                eta = (time.time() - t0) / n * (len(todo) - n) / 60
                log(f"{n}/{len(todo)} ảnh, còn ~{eta:.1f} phút")
    return out_path
