"""Bước 9 (E5, tuỳ chọn): GRPO với reward IoU, khởi tạo từ adapter SFT tốt nhất.

  python -m lab.grpo --init e1 --name e5 --config configs/e5.yaml
  python -m lab.run predict --name e5 && python -m lab.run score --name e5
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

from lab import vlm
from lab.labels import build_prompt
from lab.reward import iou_reward
from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes


def build_dataset(entries, data_dir, coord="norm1000"):
    from datasets import Dataset
    assert coord == "norm1000", "reward so khớp box trên thang 0–1000"
    rows = {"prompt": [], "image": [], "gt": []}
    for e in entries:
        rows["prompt"].append([{"role": "user", "content": build_prompt(coord, e["width"], e["height"])}])
        rows["image"].append(Image.open(Path(data_dir) / e["image_path"]).convert("RGB"))
        rows["gt"].append(json.dumps([{"label": b["label"], "bbox": list(b["bbox"])} for b in _gt_boxes(e)]))
    return Dataset.from_dict(rows)


def _text(completion):
    if isinstance(completion, list):                # prompt dạng hội thoại → completion là list message
        return "".join(m["content"] if isinstance(m["content"], str) else
                       "".join(p.get("text", "") for p in m["content"]) for m in completion)
    return completion


def iou_reward_func(completions, gt, **kwargs):
    return [iou_reward(_text(c), json.loads(g)) for c, g in zip(completions, gt)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", required=True, help="thí nghiệm SFT để khởi tạo (vd e1)")
    ap.add_argument("--name", default="e5")
    ap.add_argument("--config", default="configs/e5.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--work", default="work")
    ap.add_argument("--max-steps", type=int, help="ghi đè grpo.max_steps (gỡ lỗi)")
    ap.add_argument("--model-id", help="ghi đè model gốc (chỉ dùng để chạy thử)")
    a = ap.parse_args(argv)

    from peft import PeftModel
    from trl import GRPOConfig, GRPOTrainer
    g = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))["grpo"]
    init_dir, out_dir = Path(a.work) / a.init, Path(a.work) / a.name
    init_run = json.loads((init_dir / "run.json").read_text(encoding="utf-8"))
    model_id = a.model_id or init_run["cfg"]["model_id"]
    cuda = torch.cuda.is_available()

    processor = vlm.load_processor(model_id)
    model = vlm.load_model(model_id, quantize=cuda)
    model = PeftModel.from_pretrained(model, str(init_dir), is_trainable=True)
    n_extra = vlm.load_extra(model, init_dir)       # projector của E2/E3 (giữ đông lạnh trong GRPO)
    for n, p in model.named_parameters():
        p.requires_grad = "lora_" in n

    train = read_jsonl(Path(a.data) / "splits" / "train.jsonl")
    ds = build_dataset(train, a.data, init_run["coord"])
    args = GRPOConfig(
        output_dir=str(out_dir / "trainer"), max_steps=a.max_steps or g["max_steps"],
        learning_rate=g["learning_rate"], beta=g["beta"], num_generations=g["num_generations"],
        max_completion_length=g["max_completion_length"], per_device_train_batch_size=g["num_generations"],
        gradient_accumulation_steps=1, use_vllm=False, fp16=cuda, bf16=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False, logging_steps=1, save_strategy="no", report_to="none",
        seed=42, use_cpu=not cuda, log_completions=False)
    trainer = GRPOTrainer(model=model, reward_funcs=iou_reward_func, args=args,
                          train_dataset=ds, processing_class=processor)
    # trl ép tham số huấn luyện của model lượng tử hoá về bf16; T4 không có bf16 → đưa về fp32 cho GradScaler.
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60

    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(out_dir)
    if n_extra:
        shutil.copy(init_dir / "extra_trainable.safetensors", out_dir / "extra_trainable.safetensors")
    logs = [r for r in trainer.state.log_history if "reward" in r]
    (out_dir / "grpo_log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in logs), encoding="utf-8")
    run = {
        "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else 0.0,
        "train_minutes": round(minutes, 1), "coord": init_run["coord"], "init": a.init,
        "cfg": {**init_run["cfg"], "grpo": g},
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}))
    for r in logs[:3] + logs[-3:]:
        print({k: r[k] for k in r if k in ("step", "reward", "reward_std", "completions/mean_length", "loss")})


if __name__ == "__main__":
    main()
