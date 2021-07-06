from proportional_ranking.rules.general import ProportionalRanking
import numpy as np
import math


class enestrom(ProportionalRanking):
    """
    Rule described in https://arxiv.org/pdf/1907.10590.pdf
    """
    def __init__(self):
        super().__init__("Enestörm")

    def ranking(self):
        n, m = self.profile.shape
        scores = self.profile.copy()
        loads = [[] for _ in range(n)]
        ranking = []

        for i in range(1, m+1):
            quota = n / (i + 1)
            load_i = np.ones(n)
            for k in range(n):
                for total_score_i in loads[k]:
                    if quota > total_score_i:
                        load_i[k] = 0
                    else:
                        load_i[k] *= (1 - quota / total_score_i)

            scores_vec = np.maximum(0, load_i).dot(scores)

            j = -1
            max_v = -1
            for k in range(m):
                if scores_vec[k] - max_v > 0.0001:
                    j = k
                    max_v = scores_vec[k]
            ranking.append(j)

            av_score = np.sum(scores[:, j])
            for k in range(n):
                if scores[k, j]:
                    loads[k].append(av_score)

            scores[:, j] = False

        return ranking
