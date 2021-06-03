from proportional_ranking.rules.general import propRanking
import numpy as np
import math


class AV(propRanking):
    """
    Approval voting.
    """
    def __init__(self):
        super().__init__()
        self.name = "Approval Voting"

    def ranking(self):
        profile = self.profile
        scores = profile.copy()
        n, m = profile.shape
        weights = np.ones(n)
        scores_agg = weights.dot(scores)
        ranking = np.argsort(scores_agg)
        return ranking[::-1]

    def representation(self, alpha, lambd):
        return math.ceil(lambd * alpha / (2 * alpha - 1))

