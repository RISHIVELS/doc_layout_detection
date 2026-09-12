"""
These tests exist because the dataset conversion is the one place in this
project where a bug produces no error message.

If I get the coordinate conversion wrong, or fail to collapse the repeated
boxes, training still runs happily to completion and the loss still comes
down. I would only find out when my mAP was mysteriously bad and I had no
idea which of ten things to blame. So I pinned the three pure functions
before writing the conversion driver around them.
"""

import pytest

from scripts.prepare_dataset import coco_to_yolo, dedupe_annotations, is_valid_bbox


class TestCocoToYolo:
    """
    DocLayNet gives me [x, y, w, h] in absolute pixels measured from the
    top-left corner. Ultralytics wants [x_center, y_center, w, h] normalised
    to 0-1. Getting the corner-to-centre shift wrong shifts every box by half
    its own size, which is small enough to look like "the model is just a bit
    imprecise" rather than an obvious bug. Hence the explicit cases.
    """

    def test_converts_corner_origin_to_normalised_centre(self):
        # 512x512 box sitting in the top-left of a 1024px page: its centre is
        # at a quarter of the page in both axes, and it covers half the page.
        assert coco_to_yolo([0, 0, 512, 512], 1024, 1024) == (0.25, 0.25, 0.5, 0.5)

    def test_full_page_box_maps_to_centre_of_page(self):
        assert coco_to_yolo([0, 0, 1025, 1025], 1025, 1025) == (0.5, 0.5, 1.0, 1.0)

    def test_offset_box_keeps_its_offset(self):
        # A 100x100 box starting at (200, 400) on a 1000px page.
        xc, yc, w, h = coco_to_yolo([200, 400, 100, 100], 1000, 1000)
        assert (xc, yc) == (0.25, 0.45)
        assert (w, h) == (0.1, 0.1)

    def test_handles_non_square_pages(self):
        # I do not expect non-square pages from this dataset, but the function
        # should not silently assume squareness if I reuse it later.
        xc, yc, w, h = coco_to_yolo([0, 0, 100, 200], 200, 400)
        assert (xc, yc, w, h) == (0.25, 0.25, 0.5, 0.5)


class TestDedupeAnnotations:
    """
    This is the trap in this dataset and the reason I wrote tests at all.

    DocLayNet-base stores `bboxes_block` per *text line*, not per region. A
    paragraph spanning six lines appears as the same block box repeated six
    times, all with the same category. If I write those straight out as labels,
    a page with 40 real regions becomes a page with 1,100 annotations, most of
    them exact duplicates stacked on top of each other.

    Training on that does not crash. It just teaches the model a completely
    wrong prior about how many objects a page contains.
    """

    def test_collapses_line_level_repetition_of_the_same_block(self):
        bboxes = [[10, 10, 100, 50]] * 3 + [[10, 200, 100, 50]]
        categories = [9, 9, 9, 8]

        out = dedupe_annotations(bboxes, categories)

        assert len(out) == 2
        assert ((10.0, 10.0, 100.0, 50.0), 9) in out

    def test_keeps_identical_boxes_that_carry_different_categories(self):
        # Rare, but I do not want dedupe to silently drop a real annotation
        # just because two classes were labelled over the same extent.
        out = dedupe_annotations([[1, 1, 5, 5], [1, 1, 5, 5]], [9, 10])
        assert len(out) == 2

    def test_preserves_first_seen_order(self):
        # Order stability matters only so my label files are reproducible
        # byte-for-byte between runs, which makes diffing prep changes easy.
        out = dedupe_annotations([[5, 5, 1, 1], [1, 1, 1, 1], [5, 5, 1, 1]], [1, 2, 1])
        assert [category for _, category in out] == [1, 2]

    def test_empty_page_yields_nothing_rather_than_raising(self):
        assert dedupe_annotations([], []) == []

    def test_rejects_mismatched_input_lengths(self):
        # If these two ever come back different lengths, the dataset is not
        # what I think it is and I want to know immediately, not silently
        # truncate to the shorter one.
        with pytest.raises(ValueError):
            dedupe_annotations([[1, 1, 5, 5]], [9, 10])


class TestIsValidBbox:
    """
    Real datasets contain degenerate boxes. Ultralytics will happily accept a
    zero-area label and then produce NaN losses several epochs later, which is
    a miserable thing to debug at 2am. I drop them at prep time and count how
    many I dropped, so the number goes in the report rather than disappearing.
    """

    @pytest.mark.parametrize(
        "bbox,expected,why",
        [
            ([0, 0, 100, 100], True, "ordinary box"),
            ([0, 0, 1025, 1025], True, "exactly full page is legitimate"),
            ([0, 0, 0, 100], False, "zero width"),
            ([0, 0, 100, 0], False, "zero height"),
            ([0, 0, -10, 100], False, "negative width"),
            ([-5, 0, 100, 100], False, "starts off the left edge"),
            ([0, -5, 100, 100], False, "starts off the top edge"),
            ([1000, 0, 100, 100], False, "runs off the right edge"),
            ([0, 1000, 100, 100], False, "runs off the bottom edge"),
        ],
    )
    def test_validity_rules(self, bbox, expected, why):
        assert is_valid_bbox(bbox, 1025, 1025) is expected, why
