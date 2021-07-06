from proportional_ranking.rules.general import ProportionalRanking
import numpy as np


class sumLoads(ProportionalRanking):
    def __init__(self):
        super().__init__()
        self.name = "sumLoads"

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scores = profile.copy()
        load = np.zeros(n)
        ranking = []

        for i in range(1, m+1):
            quota = n / (i + 1)
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

