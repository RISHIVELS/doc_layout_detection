"""
I cannot exercise real inference here without a GPU and the trained weights, so
these tests are deliberately narrow: they check the two things that are pure
Python and do not need a model - lazy loading and the missing-weights error.
The real predict() path gets exercised manually against the Kaggle output and
in the API tests via a stub, not here.
"""

from app.detector import Detector


def test_is_not_loaded_before_first_use():
    detector = Detector(weights_path="nonexistent/weights.pt")
    assert detector.is_loaded is False


def test_missing_weights_file_raises_a_readable_error_not_a_torch_traceback():
    """
    Ultralytics' own error for a missing checkpoint is a stack trace from deep
    inside torch.load. That is a bad first thing to see when the actual
    problem is "you forgot to set MODEL_PATH", so I check for the file myself
    and raise something that says that plainly.
    """
    detector = Detector(weights_path="definitely/does/not/exist.pt")

    try:
        detector.load()
        assert False, "expected FileNotFoundError"
    except FileNotFoundError as error:
        assert "definitely/does/not/exist.pt" in str(error)
        assert "MODEL_PATH" in str(error)


def test_query_budget_defaults_to_the_documented_rtdetr_limit():
    from app.constants import DEFAULT_QUERY_BUDGET

    assert Detector().query_budget == DEFAULT_QUERY_BUDGET
