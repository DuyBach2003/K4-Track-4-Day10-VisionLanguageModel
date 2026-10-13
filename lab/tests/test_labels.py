import json
from pathlib import Path

import pytest

from lab.labels import box_to_pixel, build_prompt, objects_to_answer, yolo_to_box
from scripts.labcommon import read_jsonl
from scripts.score import evaluate

SPLITS = Path(__file__).resolve().parents[2] / "data" / "splits"


def test_yolo_to_box_norm1000():
    assert yolo_to_box("crack", [0.5, 0.5, 0.2, 0.1]) == {"label": "crack", "bbox_2d": [400, 450, 600, 550]}


def test_yolo_to_box_pixel():
    b = yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.1], coord="pixel", width=640, height=360)
    assert b["bbox_2d"] == [256, 162, 384, 198]


def test_yolo_to_box_clamps_and_ints():
    b = yolo_to_box("manhole", [0.02, 0.99, 0.1, 0.1])["bbox_2d"]
    assert b == [0, 940, 70, 1000]
    assert all(isinstance(v, int) for v in b)


def test_answer_sorted_top_to_bottom_then_left_to_right():
    objs = [{"label": "crack", "yolo": [0.8, 0.5, 0.1, 0.1]},
            {"label": "pothole", "yolo": [0.2, 0.5, 0.1, 0.1]},
            {"label": "manhole", "yolo": [0.5, 0.1, 0.1, 0.1]}]
    labels = [o["label"] for o in json.loads(objects_to_answer(objs))]
    assert labels == ["manhole", "pothole", "crack"]


def test_answer_compact_and_empty():
    assert objects_to_answer([]) == "[]"
    assert " " not in objects_to_answer([{"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.1]}])


def test_box_to_pixel_roundtrip():
    assert box_to_pixel([400, 450, 600, 550]) == pytest.approx([256, 162, 384, 198])
    assert box_to_pixel([1, 2, 3, 4], coord="pixel") == [1, 2, 3, 4]


def test_prompt_mentions_classes_and_scale():
    p = build_prompt()
    assert all(c in p for c in ("pothole", "crack", "manhole")) and "0-1000" in p
    assert "640x360" in build_prompt("pixel")
    with pytest.raises(ValueError):
        build_prompt("bogus")


@pytest.mark.skipif(not (SPLITS / "test.jsonl").exists(), reason="chưa chạy prepare_data.py")
@pytest.mark.parametrize("label_coord,score_coord,expect_perfect", [
    ("norm1000", "norm1000", True), ("pixel", "pixel", True),
    ("pixel", "norm1000", False), ("norm1000", "pixel", False),
])
def test_oracle_answers_through_scorer(label_coord, score_coord, expect_perfect):
    """Nhãn đúng chấm đúng thang → F1 = 1; lệch thang → F1 sụp."""
    entries = read_jsonl(SPLITS / "test.jsonl")
    preds = {e["image_id"]: objects_to_answer(e["objects"], label_coord, e["width"], e["height"])
             for e in entries}
    res = evaluate(entries, preds, coord=score_coord, n_boot=10)
    assert res["parse_valid_rate"] == 1.0
    if expect_perfect:
        assert res["micro"]["f1"] > 0.99
    else:
        assert res["micro"]["f1"] < 0.5
