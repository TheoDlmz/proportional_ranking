from proportional_ranking.rules.general import propRanking
import numpy as np
import math


class phragmen(propRanking):
    """
    This is Phragmen's rule.
    """

    def __init__(self):
        super().__init__()
        self.name = "(Seq-)Phragmen"

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scores = profile.copy()
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



