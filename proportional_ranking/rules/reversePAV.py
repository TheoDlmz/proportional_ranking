from proportional_ranking.rules.general import propRanking
import numpy as np


class reversePAV(propRanking):
    """
    reversePAV rule. Alpha parameter change weight to favors big or small groups
    (depending on the sign of alpha)
    """

    def __init__(self, alpha=0):
        super().__init__()
        self.alpha = alpha
        if alpha == 0:
            self.name = "reversePAV"
        else:
            self.name = "reversePAV (alpha = %.2f)" % alpha

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scores = profile.copy()
        weights = profile.sum(axis=1) + self.alpha
        ranking = []

        for _ in range(m):
            curr_weights = np.zeros(len(weights))

            for i in range(len(weights)):
                curr_weights[i] = max(0.01, weights[i])  # To avoid dividing by 0.

            s = (1 / curr_weights).dot(scores)
            j = -1
            min_v = np.infty
            for k in range(m):
                if (m - 1 - k) not in ranking:
                    if j == -1 or (min_v - s[m - 1 - k]) > 0.0001:
                        j = m - 1 - k
                        min_v = s[m - 1 - k]

            ranking.append(j)

            for k in range(n):
                if scores[k, j]:
                    weights[k] -= 1

            scores[:, j] = False

        return ranking[::-1]

