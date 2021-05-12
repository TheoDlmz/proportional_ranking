from proportional_ranking.rules.general import propRanking
import numpy as np
import math


class seqPAV(propRanking):

    def __init__(self, alpha=0):
        super().__init__()
        if alpha == 0:
            self.name = "seqPAV"
        else:
            self.name = "seqPAV (alpha = %.2f)" % alpha
        self.alpha = alpha

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scores = profile.copy()
        weights = np.ones(n)
        ranking = []
        for _ in range(m):
            s = (1 / (weights + self.alpha)).dot(scores)
            j = np.argmax(s)
            ranking.append(j)
            for i in range(n):
                if scores[i, j]:
                    weights[i] += 1
            scores[:, j] = False

        return ranking

    def name(self):
        return "seqPAV"

    def representation(self, alpha, lambd):
        if self.alpha == 0:
            return math.ceil(2 * (lambd + 1) ** 2 / alpha ** 2)
        else:
            raise NotImplementedError



