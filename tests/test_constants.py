"""Class vocabulary is the contract between the dataset, the model and the API.

If these drift, labels silently map to the wrong class and every downstream
metric is wrong while looking plausible. Hence the tests.
"""

from app.constants import CLASS_NAMES, CLASS_TO_ID, ID_TO_CLASS, MODEL_VERSION


def test_eleven_classes_in_fixed_alphabetical_order():
    assert CLASS_NAMES == [
        "Caption",
        "Footnote",
        "Formula",
        "List-item",
        "Page-footer",
        "Page-header",
        "Picture",
        "Section-header",
        "Table",
        "Text",
        "Title",
    ]


def test_no_class_is_a_coco_class():
    """Hard Constraint 3: at least one non-COCO class. Ours are all non-COCO."""
    coco_sample = {"person", "car", "dog", "chair", "bottle", "tv", "book"}
    assert not {c.lower() for c in CLASS_NAMES} & coco_sample


def test_id_maps_are_inverses():
    assert all(ID_TO_CLASS[CLASS_TO_ID[c]] == c for c in CLASS_NAMES)
    assert CLASS_TO_ID["Caption"] == 0
    assert CLASS_TO_ID["Title"] == 10


def test_ids_are_contiguous_from_zero():
    assert sorted(ID_TO_CLASS) == list(range(11))


def test_model_version_is_set():
    assert MODEL_VERSION and isinstance(MODEL_VERSION, str)
