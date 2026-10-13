"""Ảnh minh hoạ: box đúng (xanh lá) và box dự đoán (đỏ), kèm TP/FP/FN từng ảnh.
Chú thích trong ảnh viết không dấu vì font mặc định của PIL không có tiếng Việt.

  python -m lab.viz data      --out report/figures/data_samples.png
  python -m lab.viz pred --name e1 --out report/figures/e1_examples.png
"""
import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from lab.labels import box_to_pixel
from scripts.labcommon import read_jsonl
from scripts.score import _gt_boxes, _match, parse_output

COLORS = {"gt": (40, 220, 90), "pred": (235, 45, 45)}
CAPTION_H = 38


def _draw_boxes(draw, boxes, color, w, h, coord="norm1000", width=2):
    for b in boxes:
        x1, y1, x2, y2 = box_to_pixel(b["bbox"], coord, w, h)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        draw.text((x1 + 2, max(0, y1 - 11)), b["label"], fill=color)


def render(entry, data_dir, pred=None, coord="norm1000", caption=""):
    img = Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")
    w, h = img.size
    canvas = Image.new("RGB", (w, h + CAPTION_H), (255, 255, 255))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    _draw_boxes(d, _gt_boxes(entry), COLORS["gt"], w, h)
    if pred is not None:
        _draw_boxes(d, pred, COLORS["pred"], w, h, coord)
    for i, line in enumerate(caption.split("\n")[:2]):
        d.text((6, h + 4 + 16 * i), line, fill=(0, 0, 0))
    return canvas


def grid(tiles, cols=2):
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), (255, 255, 255))
    for i, t in enumerate(tiles):
        out.paste(t, ((i % cols) * (w + 6), (i // cols) * (h + 6)))
    return out


def per_image(entries, preds, coord="norm1000"):
    """[(entry, pred_boxes_in_native_coord, tp, fp, fn, raw)] cho mỗi ảnh."""
    from scripts.score import _to_norm1000
    rows = []
    for e in entries:
        raw = preds.get(e["image_id"])
        parsed = parse_output(raw) if raw is not None else None
        pred = parsed or []
        per, _ = _match(_gt_boxes(e), _to_norm1000(pred, coord, e["width"], e["height"]))
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        rows.append((e, pred, tp, fp, fn, raw if parsed is None else None))
    return rows


def pick_examples(rows, n_good=3, n_bad=3):
    """Ảnh đúng nhiều nhất và ảnh sai nhiều nhất (theo FP+FN)."""
    good = sorted((r for r in rows if r[2] > 0), key=lambda r: (r[3] + r[4], -r[2]))[:n_good]
    used = {id(r) for r in good}
    bad = sorted((r for r in rows if id(r) not in used), key=lambda r: -(r[3] + r[4]))[:n_bad]
    return good + bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["data", "pred"])
    ap.add_argument("--name", help="tên thí nghiệm (đọc work/<name>_<split>.jsonl)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--coord", default="norm1000")
    ap.add_argument("--data", default="data")
    ap.add_argument("--work", default="work")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.mode == "data":
        entries = read_jsonl(Path(a.data) / "splits" / "train.jsonl")
        step = max(1, len(entries) // a.n)
        tiles = [render(e, a.data, caption=f"{e['image_id']} ({e['session']}): "
                        + ", ".join(o["label"] for o in e["objects"]))
                 for e in entries[::step][:a.n]]
    else:
        entries = read_jsonl(Path(a.data) / "splits" / f"{a.split}.jsonl")
        preds = {r["image_id"]: r["raw"] for r in read_jsonl(Path(a.work) / f"{a.name}_{a.split}.jsonl")}
        chosen = pick_examples(per_image(entries, preds, a.coord), a.n // 2, a.n - a.n // 2)
        tiles = []
        for e, pred, tp, fp, fn, bad_raw in chosen:
            cap = f"{a.name.upper()} {e['image_id']}: TP={tp} FP={fp} FN={fn}"
            cap += "\nbroken JSON: " + bad_raw[:70] if bad_raw is not None else \
                "\ngreen = ground truth, red = prediction"
            tiles.append(render(e, a.data, pred, a.coord, cap))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid(tiles).save(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
