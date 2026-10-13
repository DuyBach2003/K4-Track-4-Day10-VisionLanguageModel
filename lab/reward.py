"""Bước 9 (E5): reward IoU kiểm chứng được cho GRPO. Python thuần, không cần GPU."""
from scripts.score import IOU_THR, iou_xyxy, parse_output


def match_iou(gt, pred, thr=IOU_THR):
    """Ghép đôi tham lam theo IoU giảm dần, chỉ cùng lớp, chỉ nhận IoU ≥ thr. Trả list IoU các cặp."""
    pairs = sorted(((iou_xyxy(g["bbox"], p["bbox"]), i, j)
                    for i, g in enumerate(gt) for j, p in enumerate(pred)
                    if g["label"] == p["label"]), reverse=True)
    used_g, used_p, ious = set(), set(), []
    for v, i, j in pairs:
        if v < thr:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        ious.append(v)
    return ious


def iou_reward(completion, gt_boxes):
    """gt_boxes: list[{"label", "bbox": (x1,y1,x2,y2)}] cùng thang với completion.

    0.0 nếu JSON hỏng; 1.0 nếu cả gt lẫn pred rỗng;
    còn lại 0.2 + 0.8 * tổng IoU khớp / max(n_gt, n_pred, 1) — box bịa bị phạt qua mẫu số.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt_boxes and not pred:
        return 1.0
    total = sum(match_iou(gt_boxes, pred))
    return 0.2 + 0.8 * total / max(len(gt_boxes), len(pred), 1)
