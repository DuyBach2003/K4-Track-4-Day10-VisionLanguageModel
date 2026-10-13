"""Phân tích thêm cho REPORT từ dự đoán thô trong report/runs/: bootstrap ghép cặp, F1 theo phiên,
kiểu lỗi E0, kiểu FP, chấm sai thang, recall theo kích thước box, dấu hiệu reward hacking của E5.

  python -m lab.analyze
"""
import collections
import json
import random
import re
import statistics
from pathlib import Path

from scripts.labcommon import CLASSES, read_jsonl
from scripts.score import _gt_boxes, _match, _prf, _to_norm1000, evaluate, iou_xyxy, parse_output

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "report" / "runs"
NAMES = ("e0", "e1", "e2", "e3", "e4_coord_scale_pixel", "e5")
SIZE_BINS = [(0, 0.3), (0.3, 1), (1, 3), (3, 100)]


def coord_of(name):
    return "pixel" if "pixel" in name else "norm1000"


def pred_boxes(raw, coord, e):
    return _to_norm1000(parse_output(raw) or [], coord, e["width"], e["height"])


def image_counts(entries, preds, coord):
    rows = []
    for e in entries:
        per, _ = _match(_gt_boxes(e), pred_boxes(preds.get(e["image_id"]), coord, e))
        rows.append(tuple(sum(v[k] for v in per.values()) for k in range(3)))
    return rows


def f1(rows):
    return _prf(*(sum(r[k] for r in rows) for k in range(3)))["f1"]


def paired_bootstrap(ra, rb, n_boot=2000, seed=0):
    """ΔF1 = F1(a) − F1(b) trên cùng các mẫu ảnh lấy lại. Trả (Δ, lo, hi, P(Δ ≤ 0))."""
    rng, n, d = random.Random(seed), len(ra), []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        d.append(f1([ra[i] for i in idx]) - f1([rb[i] for i in idx]))
    d.sort()
    return (f1(ra) - f1(rb), d[int(0.025 * n_boot)], d[int(0.975 * n_boot) - 1],
            sum(x <= 0 for x in d) / n_boot)


def fp_types(entries, preds, coord):
    """Phân loại box dự đoán không khớp: sai lớp / lệch vị trí (IoU 0,1–0,5) / bịa (IoU < 0,1)."""
    out = collections.Counter()
    for e in entries:
        gt = _gt_boxes(e)
        for x in pred_boxes(preds.get(e["image_id"]), coord, e):
            same = [iou_xyxy(x["bbox"], y["bbox"]) for y in gt if y["label"] == x["label"]]
            other = [iou_xyxy(x["bbox"], y["bbox"]) for y in gt if y["label"] != x["label"]]
            if same and max(same) >= 0.5:
                continue
            if other and max(other) >= 0.5:
                out["sai lớp"] += 1
            elif same + other and max(same + other) >= 0.1:
                out["lệch vị trí"] += 1
            else:
                out["bịa"] += 1
    return dict(out)


def recall_by_size(entries, preds, coord):
    res = {}
    for cls in CLASSES:
        hit, tot = collections.Counter(), collections.Counter()
        for e in entries:
            p = pred_boxes(preds.get(e["image_id"]), coord, e)
            for o, g in zip(e["objects"], _gt_boxes(e)):
                if o["label"] != cls:
                    continue
                area = o["yolo"][2] * o["yolo"][3] * 100
                b = next(i for i, (lo, hi) in enumerate(SIZE_BINS) if lo <= area < hi)
                tot[b] += 1
                hit[b] += any(x["label"] == cls and iou_xyxy(x["bbox"], g["bbox"]) >= 0.5 for x in p)
        res[cls] = {f"{SIZE_BINS[b][0]}-{SIZE_BINS[b][1]}%": f"{hit[b]}/{tot[b]}" for b in sorted(tot)}
    return res


def frame_time(image_id):
    """Giây trong ngày lấy từ tên file (vlcsnap-…-HHhMMmSSsmmm hoặc YYYYMMDD_HHMMSS); None nếu không có."""
    m = re.match(r"vlcsnap-\d{4}-\d{2}-\d{2}-(\d{2})h(\d{2})m(\d{2})s(\d+)", image_id)
    if m:
        h, mi, sec, ms = map(int, m.groups())
        return h * 3600 + mi * 60 + sec + ms / 1000
    m = re.match(r"\d{8}_(\d{2})(\d{2})(\d{2})", image_id)
    if m:
        h, mi, sec = map(int, m.groups())
        return h * 3600 + mi * 60 + sec
    return None


def leakage(entries_by_split, seed=0):
    """Ảnh test có frame train cùng phiên cách ≤ 5 s: split ngẫu nhiên theo ảnh so với split theo phiên."""
    entries = [e for es in entries_by_split.values() for e in es]
    timed = {e["image_id"]: (e["session"], frame_time(e["image_id"])) for e in entries
             if frame_time(e["image_id"]) is not None}
    ids = [e["image_id"] for e in entries]
    random.Random(seed).shuffle(ids)
    n = len(ids)
    splits = {"ngẫu nhiên 70/15/15": (set(ids[:int(0.7 * n)]), set(ids[int(0.85 * n):])),
              "theo phiên (đề)": ({e["image_id"] for e in entries_by_split["train"]},
                                  {e["image_id"] for e in entries_by_split["test"]})}
    for name, (train, test) in splits.items():
        gaps = []
        for i in test & timed.keys():
            sess, t = timed[i]
            near = [abs(t - tt) for j, (ss, tt) in timed.items() if j in train and ss == sess]
            gaps.append(min(near) if near else float("inf"))
        close = sum(g <= 5 for g in gaps)
        print(f"  {name}: {close}/{len(gaps)} ảnh test ({close / len(gaps):.0%}) có frame train cùng phiên cách ≤ 5 s")


def main():
    by_split = {s: read_jsonl(ROOT / "data" / "splits" / f"{s}.jsonl") for s in ("train", "val", "test")}
    print("Rò rỉ theo thời gian:")
    leakage(by_split)
    entries = by_split["test"]
    preds = {n: {r["image_id"]: r["raw"] for r in read_jsonl(RUNS / f"{n}_test.jsonl")}
             for n in NAMES if (RUNS / f"{n}_test.jsonl").exists()}
    rows = {n: image_counts(entries, p, coord_of(n)) for n, p in preds.items()}
    print("F1:", {n: round(f1(r), 3) for n, r in rows.items()})

    print("\nBootstrap ghép cặp (ΔF1, CI95 thấp, CI95 cao, P(Δ≤0)):")
    for a, b in [("e1", "e0"), ("e2", "e1"), ("e3", "e2"), ("e3", "e1"),
                 ("e3", "e4_coord_scale_pixel"), ("e5", "e3")]:
        if a in rows and b in rows:
            print(f"  {a} − {b}:", tuple(round(v, 3) for v in paired_bootstrap(rows[a], rows[b])))

    print("\nF1 theo phiên (số ảnh, F1):")
    sess = collections.defaultdict(list)
    for i, e in enumerate(entries):
        sess[e["session"]].append(i)
    for n in ("e1", "e2", "e3"):
        print(f"  {n}", {s: (len(ix), round(f1([rows[n][i] for i in ix]), 3)) for s, ix in sess.items()})

    raws = [preds["e0"][e["image_id"]] for e in entries]
    bad = [r for r in raws if parse_output(r) is None]
    print(f"\nE0: {len(bad)} output hỏng; box/ảnh dự đoán "
          f"{sum(len(parse_output(r) or []) for r in raws) / len(raws):.2f} so với nhãn "
          f"{sum(len(e['objects']) for e in entries) / len(entries):.2f}; "
          f"toạ độ > 1000: {sum(1 for r in raws for b in (parse_output(r) or []) if max(b['bbox']) > 1000)}; "
          f"có ```: {sum('```' in r for r in raws)}")
    print("  không đóng ngoặc (bị cắt):", sum(not r.strip().endswith("]") for r in bad))

    for n in ("e0", "e3", "e5"):
        if n in preds:
            print(f"Kiểu FP {n}:", fp_types(entries, preds[n], coord_of(n)))

    for n, wrong in (("e3", "pixel"), ("e4_coord_scale_pixel", "norm1000")):
        if n in preds:
            r = evaluate(entries, preds[n], coord=wrong, n_boot=50)
            print(f"Chấm {n} sai thang ({wrong}): F1={r['micro']['f1']:.3f} valid={r['parse_valid_rate']:.3f}")

    print("\nRecall E3 theo diện tích box:", json.dumps(recall_by_size(entries, preds["e3"], "norm1000"),
                                                       ensure_ascii=False))

    if "e5" in preds:
        print("\nDấu hiệu reward hacking trên test:")
        for n in ("e3", "e5"):
            ps = [parse_output(preds[n][e["image_id"]]) for e in entries]
            big = sum(1 for p in ps if p for b in p
                      if (b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]) > 0.25e6)
            print(f"  {n}: trả [] {sum(p == [] for p in ps)} | box > 25% khung hình {big} | "
                  f"box/ảnh {sum(len(p or []) for p in ps) / len(ps):.2f}")
        log = read_jsonl(RUNS / "e5" / "grpo_log.jsonl")
        for key in ("reward", "reward_std", "completions/mean_length", "completions/clipped_ratio",
                    "frac_reward_zero_std"):
            v = [r[key] for r in log if key in r]
            print(f"  {key}: 10 bước đầu {statistics.mean(v[:10]):.3f} | 10 bước cuối "
                  f"{statistics.mean(v[-10:]):.3f} | trung bình {statistics.mean(v):.3f}")


if __name__ == "__main__":
    main()
