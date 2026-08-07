from PIL import Image, ImageDraw

from reprforge.representation_granularity import (
    aggregate_unit_ranking,
    deterministic_neutral_order,
    fixed_quadrants,
    parent_metrics,
    xycut_regions,
)


def test_fixed_quadrants_cover_page_with_overlap():
    boxes = fixed_quadrants(100, 200, overlap_fraction=0.1)
    assert boxes == [(0, 0, 55, 110), (45, 0, 100, 110), (0, 90, 55, 200), (45, 90, 100, 200)]


def test_xycut_splits_two_content_bands():
    image = Image.new("L", (100, 100), 255)
    draw = ImageDraw.Draw(image)
    draw.rectangle((10, 10, 90, 35), fill=0)
    draw.rectangle((10, 65, 90, 90), fill=0)
    boxes = xycut_regions(
        image,
        maximum_units=2,
        analysis_maximum_side=100,
        ink_threshold=245,
        minimum_region_fraction=0.1,
        minimum_whitespace_gap_fraction=0.02,
        crop_padding_fraction=0.0,
    )
    assert len(boxes) == 2
    assert boxes[0][3] <= boxes[1][1] + 1


def test_stable_parent_max_aggregation_and_metrics():
    parents = aggregate_unit_ranking(
        [("a0", 2.0), ("a1", 3.0), ("b0", 2.5)],
        {"a0": "a", "a1": "a", "b0": "b"},
    )
    assert parents == [("a", 3.0), ("b", 2.5)]
    assert parent_metrics(["a", "b"], {"b": 1.0}, depth=2)["hit"] is True


def test_neutral_order_is_deterministic():
    first = deterministic_neutral_order(["1", "2", "3"], protocol_id="p", domain="d")
    second = deterministic_neutral_order(["3", "2", "1"], protocol_id="p", domain="d")
    assert first == second
