from gap_scorer import Annotation, calculate_scores, make_scorecard_simple
from constants import PRONOUNS, Gender

class GAPEvaluator:

    def __init__(self):
        pass
    
    def eval(self, gold_annotations, system_annotations):
        scores = calculate_scores(gold_annotations, system_annotations)
        return make_scorecard_simple(scores)