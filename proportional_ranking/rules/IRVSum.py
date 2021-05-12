from proportional_ranking.rules.general import propRanking
import numpy as np


class IRVSum(propRanking):
    def __init__(self):
        super().__init__()
        self.name = "IRVSum"

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scores = profile.copy()
        load = np.zeros(n)
        ranking = []

        for i in range(m):
            quota = n / (i + 2)
            load_k = quota * load
            s = (1 - np.minimum(1, load_k)).dot(scores)

            j = 0
            max_v = s[0]
            for k in range(m):
                if s[k] - max_v > 0.0001:
                    j = k
                    max_v = s[j]

            ranking.append(j)

            total_score = np.sum(scores[:, j])
            for k in range(n):
                if scores[k, j]:
                    load[k] += 1 / total_score

            scores[:, j] = False

        return ranking

    def name(self):
        return "IRVsum"
