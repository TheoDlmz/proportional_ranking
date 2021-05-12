from proportional_ranking.rules.general import propRanking
import numpy as np


class geometric(propRanking):

    def __init__(self, alpha):
        super().__init__()
        self.profile = None
        self.alpha = alpha
        self.name = "geometric (%.2f)" % self.alpha

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scores = profile.copy()
        weights = np.ones(n)
        ranking = []
        for _ in range(m):
            s = (1 / (weights ** self.alpha)).dot(scores)
            j = np.argmax(s)
            ranking.append(j)
            for i in range(n):
                if scores[i, j]:
                    weights[i] += 1
            scores[:, j] = False

        return ranking

