"""Bước 2: nhãn YOLO → JSON (thang 0–1000 hoặc pixel) và prompt. Python thuần, không cần GPU."""
import json

from scripts.labcommon import CLASSES

COORD_SCALES = ("norm1000", "pixel")

_PROMPT_HEAD = (
    "You are inspecting a dashcam image of a road. Detect every instance of road damage "
    f"from exactly these classes: {', '.join(CLASSES)}.\n"
)
_PROMPT_TAIL = (
    "Answer with ONLY a JSON list, no markdown and no extra text, in the form "
    '[{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}], where (x1, y1) is the top-left '
    "and (x2, y2) the bottom-right corner. Return [] if there is no damage."
)


def build_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        scale = ("Coordinates are integers on a 0-1000 scale relative to the image "
                 "(0 = left/top edge, 1000 = right/bottom edge).\n")
    elif coord == "pixel":
        scale = (f"Coordinates are integer pixel positions in the original {width}x{height} "
                 f"image (x in 0-{width}, y in 0-{height}).\n")
    else:
        raise ValueError(f"unknown coord scale: {coord}")
    return _PROMPT_HEAD + scale + _PROMPT_TAIL


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (tâm + kích thước, chuẩn hoá [0,1]) → {"label", "bbox_2d": [x1,y1,x2,y2]} số nguyên."""
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    box = [(xc - w / 2) * sx, (yc - h / 2) * sy, (xc + w / 2) * sx, (yc + h / 2) * sy]
    lim = (sx, sy, sx, sy)
    box = [min(max(int(round(v)), 0), m) for v, m in zip(box, lim)]
    return {"label": label, "bbox_2d": box}


def objects_to_boxes(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    # trên→dưới, trái→phải: thứ tự ổn định cho model học
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0]))
    return boxes


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    return json.dumps(objects_to_boxes(objects, coord, width, height), separators=(",", ":"))


def box_to_pixel(bbox, coord="norm1000", width=640, height=360):
    """Quy box về pixel ảnh gốc để vẽ."""
    if coord == "pixel":
        return [float(v) for v in bbox]
    x1, y1, x2, y2 = bbox
    return [x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height]
