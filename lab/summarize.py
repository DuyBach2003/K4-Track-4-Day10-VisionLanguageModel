"""In bảng tổng hợp Markdown từ report/evidence/*.json để dán vào REPORT.md.

  python -m lab.summarize
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUPS = {"e0": "zero-shot", "e1": "LoRA LLM", "e2": "+ projector", "e3": "+ LoRA ViT", "e5": "GRPO từ E3"}


def row(key, d):
    m, pc, meta = d["micro"], d["per_class"], d.get("meta", {})
    setup = SETUPS.get(key, key.removeprefix("e4_").replace("_", " "))
    params = meta.get("trainable_params", 0)
    cells = [key.upper() if key in SETUPS else "E4", setup,
             f"{params / 1e6:.1f}M" if params else "0",
             f"{meta['peak_vram_gb']:.2f}" if "peak_vram_gb" in meta else "",
             f"{meta['train_minutes']:.1f}" if "train_minutes" in meta else "",
             f"{m['f1']:.3f}", f"{d['f1_ci95'][0]:.3f}–{d['f1_ci95'][1]:.3f}",
             f"{d['fp_per_image']:.2f}",
             *(f"{pc[c]['f1']:.3f}" for c in ("pothole", "crack", "manhole")),
             f"{d['parse_valid_rate']:.3f}"]
    return "| " + " | ".join(cells) + " |"


def main():
    ev = ROOT / "report" / "evidence"
    files = sorted(ev.glob("e*.json"), key=lambda p: (p.stem[:2], p.stem))
    print("| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh "
          "| F1 pothole | F1 crack | F1 manhole | parse_valid |")
    print("|" + "---|" * 12)
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("split") == "test":
            print(row(p.stem, d))


if __name__ == "__main__":
    main()
