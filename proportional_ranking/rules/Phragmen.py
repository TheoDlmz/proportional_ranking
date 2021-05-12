from proportional_ranking.rules.general import propRanking
import numpy as np
import math


class phragmenMinmax(propRanking):
    def __init__(self):
        super().__init__()
        self.name = "PhragmenMinmax"

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scores = profile.copy()
        load = np.zeros(n)
        ranking = []

        for _ in range(m):
            s = (1 + load.dot(scores)) / (np.ones(n).dot(scores))
            j = 0
            min_v = s[0]
            for k in range(m):
                if min_v - s[k] > 0.0001:
                    j = k
                    min_v = s[j]

            ranking.append(j)

            for k in range(n):
                if scores[k, j]:
                    load[k] = s[j]

            scores[:, j] = False

        return ranking

    def representation(self, alpha, lambd):
        return math.ceil(5 * lambd / alpha ** 2 + 1 / alpha)


class phragmenClassic(propRanking):
    def __init__(self):
        super().__init__()
        self.name = "PhragmenClassic"

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scores = profile.copy()
        load = np.ones(n)
        ranking = []

        for i in range(m):
            quota = n / (i + 2)
            s = np.maximum(0, load).dot(scores)

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
                    if quota > total_score:
                        load[k] *= 0
                    else:
                        load[k] *= (1 - quota / total_score)

            scores[:, j] = False

        return ranking


class phragmenDepile(propRanking):
    def __init__(self):
        super().__init__()
        self.name = "PhragmenDepile"

    def ranking(self):
        profile = self.profile
        n, m = profile.shape
        scores = profile.copy()
        load = [[] for _ in range(n)]
        ranking = []

        for i in range(m):
            quota = n / (i + 2)
            load_i = np.ones(n)
            for k in range(n):
                for total_score_i in load[k]:
                    if quota > total_score_i:
                        load_i[k] = 0
                    else:
                        load_i[k] *= (1 - quota / total_score_i)

            s = np.maximum(0, load_i).dot(scores)

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
                    load[k].append(total_score)

            scores[:, j] = False

        return ranking
