"""
The failure ranking decides which five pages end up in my memo, so I wanted the
scoring to be something I had checked rather than something I hoped was right.

The distinction I care about most is between a *miss* and a *misclassification*.
They look similar in an aggregate metric and they have completely different root
causes - one is a vision failure, the other is usually genuine ambiguity in the
labelling protocol. If the scoring conflates them, my failure analysis ends up
saying "the model missed things" instead of saying something useful.
"""

from scripts.mine_failures import iou, score_page


class TestIou:
    def test_identical_boxes_overlap_completely(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes_do_not_overlap(self):
        assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_touching_edges_count_as_no_overlap(self):
        assert iou((0, 0, 10, 10), (10, 0, 20, 10)) == 0.0

    def test_half_overlap(self):
        # Two 10x10 boxes sharing a 5x10 strip: intersection 50, union 150.
        assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == 50 / 150


class TestScorePage:
    def test_perfect_prediction_scores_zero_error(self):
        gt = [(8, (0, 0, 10, 10)), (9, (20, 20, 30, 30))]
        pred = [(8, (0, 0, 10, 10), 0.9), (9, (20, 20, 30, 30), 0.9)]

        score = score_page(gt, pred)

        assert score["correct"] == 2
        assert score["error_score"] == 0

    def test_located_but_wrongly_labelled_counts_as_misclassified_not_missed(self):
        """
        This is the case the whole script is built around. The model found the
        region perfectly and called it the wrong thing. That is a semantics
        problem, not a detection problem, and it must not be reported as a miss.
        """
        gt = [(9, (0, 0, 10, 10))]          # Text
        pred = [(3, (0, 0, 10, 10), 0.8)]   # List-item

        score = score_page(gt, pred)

        assert score["misclassified"] == 1
        assert score["missed"] == 0
        assert score["false_positives"] == 0

    def test_region_the_model_never_found_counts_as_missed(self):
        score = score_page([(8, (0, 0, 10, 10))], [])
        assert score["missed"] == 1
        assert score["misclassified"] == 0

    def test_invented_region_counts_as_false_positive(self):
        score = score_page([], [(8, (0, 0, 10, 10), 0.7)])
        assert score["false_positives"] == 1
        assert score["missed"] == 0

    def test_poorly_localised_prediction_is_both_a_miss_and_a_false_positive(self):
        # Below the IoU threshold, so it matches nothing: the true region was
        # missed and the prediction was spurious. That is the correct reading.
        gt = [(8, (0, 0, 10, 10))]
        pred = [(8, (8, 8, 18, 18), 0.9)]

        score = score_page(gt, pred)

        assert score["missed"] == 1
        assert score["false_positives"] == 1

    def test_each_prediction_can_only_match_one_region(self):
        # Two adjacent ground-truth regions, one prediction. Only one can be
        # satisfied; the other is a genuine miss.
        gt = [(8, (0, 0, 10, 10)), (8, (0, 0, 10, 10))]
        pred = [(8, (0, 0, 10, 10), 0.9)]

        score = score_page(gt, pred)

        assert score["correct"] == 1
        assert score["missed"] == 1

    def test_misclassification_is_weighted_above_a_plain_miss(self):
        misclassified = score_page([(9, (0, 0, 10, 10))], [(3, (0, 0, 10, 10), 0.8)])
        missed = score_page([(9, (0, 0, 10, 10))], [])

        assert misclassified["error_score"] > missed["error_score"]

    def test_empty_page_predicted_empty_is_not_an_error(self):
        assert score_page([], [])["error_score"] == 0
