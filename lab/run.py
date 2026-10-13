"""CLI chạy thí nghiệm. Ví dụ (trên Colab, từ gốc repo):

  python -m lab.run predict --name e0 --zero-shot                    # E0
  python -m lab.run score   --name e0
  python -m lab.run train   --name e1 --config configs/e1.yaml       # E1 (tương tự e2, e3); bị ngắt thì chạy lại y hệt
  python -m lab.run predict --name e1
  python -m lab.run score   --name e1
  python -m lab.run train   --name e4_coord_scale_pixel --config configs/e1.yaml --set coord=pixel
  python -m lab.run score   --name e4_coord_scale_norm1000 --pred-from e1   # giá trị gốc = setup gốc

Thư mục: work/<name>/ (adapter, extra_trainable, run.json), work/<name>_<split>.jsonl (dự đoán).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.labcommon import read_jsonl

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = {"coord": "norm1000", "train_fraction": 1.0}


def _cast(v):
    try:
        return yaml.safe_load(v)
    except yaml.YAMLError:
        return v


def load_cfg(path, overrides=()):
    cfg = dict(DEFAULTS)
    cfg.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    for item in overrides:
        key, _, val = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = _cast(val)
    return cfg


def _run_json(work, name):
    p = Path(work) / name / "run.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cmd_train(a):
    from lab import vlm
    cfg = load_cfg(a.config, a.set)
    if (Path(a.work) / a.name / "run.json").exists():
        sys.exit(f"{a.work}/{a.name} đã train xong (có run.json); xoá thư mục nếu muốn train lại")
    entries = vlm.subsample(read_jsonl(Path(a.data) / "splits" / "train.jsonl"),
                            cfg["train_fraction"], cfg["seed"])
    print(f"[{a.name}] {cfg['description']} | coord={cfg['coord']} | {len(entries)} ảnh train")
    processor = vlm.load_processor(cfg["model_id"], padding_side="right")
    model = vlm.load_model(cfg["model_id"])
    model, info = vlm.apply_setup(model, cfg["lora"], cfg["train_projector"])
    print(json.dumps(info))
    out_dir = Path(a.work) / a.name
    progress = vlm.resume_checkpoint(model, out_dir)   # tự chạy tiếp nếu lần trước bị ngắt
    run = vlm.train(model, processor, entries, cfg, a.data, out_dir, coord=cfg["coord"],
                    num_workers=a.workers, save_every=a.save_every, progress=progress)
    run.update(info)
    (Path(a.work) / a.name / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({k: run[k] for k in ("trainable_params", "peak_vram_gb", "train_minutes")}))


def cmd_predict(a):
    from lab import vlm
    entries = read_jsonl(Path(a.data) / "splits" / f"{a.split}.jsonl")
    if a.limit:
        entries = entries[:a.limit]
    run = None if a.zero_shot else _run_json(a.work, a.name)
    if not a.zero_shot and run is None:
        sys.exit(f"không thấy {a.work}/{a.name}/run.json; train trước hoặc dùng --zero-shot")
    coord = run["coord"] if run else "norm1000"
    model_id = run["cfg"]["model_id"] if run else vlm.MODEL_ID
    processor = vlm.load_processor(model_id)
    model, n_extra = vlm.load_for_inference(None if a.zero_shot else Path(a.work) / a.name, model_id)
    if run and run["cfg"].get("train_projector") and n_extra == 0:
        sys.exit("setup có train projector nhưng không nạp được extra_trainable.safetensors")
    print(f"[{a.name}] coord={coord} extra_tensors={n_extra} n={len(entries)}")
    out = Path(a.work) / f"{a.name}_{a.split}.jsonl"
    vlm.predict(model, processor, entries, a.data, out, coord=coord,
                batch_size=a.batch_size, max_new_tokens=a.max_new_tokens)
    for r in read_jsonl(out)[:5]:
        print(r["image_id"], "→", r["raw"][:200])


def cmd_score(a):
    src = a.pred_from or a.name
    run = _run_json(a.work, src)
    pred = Path(a.work) / f"{src}_{a.split}.jsonl"
    out = Path(a.out) if a.out else ROOT / "report" / "evidence" / (
        f"{a.name}.json" if a.split == "test" else f"{a.name}_{a.split}.json")
    cmd = [sys.executable, str(ROOT / "scripts" / "score.py"), "--pred", str(pred),
           "--splits-dir", str(Path(a.data) / "splits"), "--split", a.split,
           "--name", a.name.upper(), "--out", str(out),
           "--coord", run["coord"] if run else "norm1000"]
    if run:
        for k in ("trainable_params", "peak_vram_gb", "train_minutes"):
            cmd += ["--meta", f"{k}={run[k]}"]
    n_pred, n_split = len(read_jsonl(pred)), len(read_jsonl(Path(a.data) / "splits" / f"{a.split}.jsonl"))
    if n_pred < n_split:
        print(f"CHÚ Ý: chỉ có {n_pred}/{n_split} dự đoán; ảnh thiếu bị tính là bỏ sót (chỉ dùng để gỡ lỗi)")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--work", default="work")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--config", required=True)
    t.add_argument("--set", action="append", default=[], help="ghi đè cfg, vd lora.r=4, coord=pixel")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--save-every", type=int, default=100, help="lưu checkpoint mỗi N bước")
    t.set_defaults(fn=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--name", required=True)
    p.add_argument("--zero-shot", action="store_true")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (gỡ lỗi trên val)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.set_defaults(fn=cmd_predict)

    s = sub.add_parser("score")
    s.add_argument("--name", required=True)
    s.add_argument("--pred-from", help="dùng dự đoán + run.json của thí nghiệm khác (giá trị gốc của E4)")
    s.add_argument("--split", default="test", choices=["val", "test"])
    s.add_argument("--out")
    s.set_defaults(fn=cmd_score)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
