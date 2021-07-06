from proportional_ranking.rules.general import ProportionalRanking
import numpy as np
import math
from fractions import Fraction


class Phragmen(ProportionalRanking):
    """
    Phragmen's voting rule. See paper Proportional Ranking by Skowron et Al for more details.


    """

    def __init__(self):
        super().__init__("Phragmen")

    def ranking(self):
        n, m = self.profile.shape

        scores = self.profile.copy()
        load = np.zeros(n)
        ranking = []

        for _ in range(m):
            j = -1
            min_v = np.infty
            for k in range(m):
                av_score = scores[:, k].sum()
                if av_score == 0:
                    if min_v == np.infty and k not in ranking:
                        j = k
                else:
                    s = 1 + load.dot(scores[:, k])
                    s /= av_score
                    if min_v - s > 0.0001:
                        j = k
                        min_v = s

            ranking.append(j)

            for k in range(n):
                if scores[k, j]:
                    load[k] = min_v

            scores[:, j] = False

        return ranking

    def representation(self, alpha, lambd):
        return math.ceil(5 * lambd / alpha ** 2 + 1 / alpha)


