import json

import pytest

from lab.reward import iou_reward

GT = [{"label": "pothole", "bbox": (100, 100, 200, 200)},
      {"label": "crack", "bbox": (500, 500, 700, 600)}]


def ans(*boxes):
    return json.dumps([{"label": lab, "bbox_2d": list(bb)} for lab, bb in boxes])


def test_perfect():
    assert iou_reward(ans(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))), GT) == 1.0


def test_perfect_in_markdown_fence():
    raw = "```json\n" + ans(("pothole", (100, 100, 200, 200)), ("crack", (500, 500, 700, 600))) + "\n```"
    assert iou_reward(raw, GT) == 1.0


@pytest.mark.parametrize("raw", ["", "not json", '[{"label": "crack"}]', '{"label": "crack"}',
                                 '[{"label": "crack", "bbox_2d": [5, 5, 1, 1]}]'])
def test_broken_json_is_zero(raw):
    assert iou_reward(raw, GT) == 0.0


def test_wrong_class_gets_only_format_reward():
    assert iou_reward(ans(("manhole", (100, 100, 200, 200)), ("manhole", (500, 500, 700, 600))), GT) == 0.2


def test_extra_box_is_penalised_via_denominator():
    one = [GT[0]]
    exact = iou_reward(ans(("pothole", (100, 100, 200, 200))), one)
    extra = iou_reward(ans(("pothole", (100, 100, 200, 200)), ("crack", (800, 800, 900, 900))), one)
    assert exact == 1.0
    assert extra == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_empty_prediction_with_gt():
    assert iou_reward("[]", GT) == 0.2


def test_empty_both_is_full_reward():
    assert iou_reward("[]", []) == 1.0


def test_hallucination_on_empty_image():
    assert iou_reward(ans(("crack", (0, 0, 10, 10))), []) == 0.2


def test_giant_box_does_not_hack_reward():
    assert iou_reward(ans(("pothole", (0, 0, 1000, 1000)), ("crack", (0, 0, 1000, 1000))), GT) == 0.2


def test_partial_iou_counts_only_above_threshold():
    # IoU = 0.5 đúng ngưỡng: được tính; IoU < 0.5: không
    half = iou_reward(ans(("pothole", (100, 100, 200, 150))), [GT[0]])
    assert half == pytest.approx(0.2 + 0.8 * 0.5)
    low = iou_reward(ans(("pothole", (100, 100, 200, 140))), [GT[0]])
    assert low == pytest.approx(0.2)


def test_each_gt_matched_once():
    dup = ans(("pothole", (100, 100, 200, 200)), ("pothole", (100, 100, 200, 200)))
    assert iou_reward(dup, [GT[0]]) == pytest.approx(0.2 + 0.8 * 1 / 2)
