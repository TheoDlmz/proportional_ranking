from proportional_ranking.rules.general import propRanking
from itertools import permutations


def bordaPAV_score(profile, ranking):
    m = len(ranking)
    s_tot = 0
    for approval in profile:
        s = 0
        div = 1
        for i in range(m):
            if approval[ranking[i]]:
                s += (m - 1 - i) / div
                div += 1
        s_tot += s
    return s_tot


class bordaPAV(propRanking):
    def __init__(self):
        super().__init__()
        self.name = "bordaPAV"

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        score_max = 0
        best = None
        for ranking in list(permutations(range(m))):
            new_score = bordaPAV_score(profile, ranking)
            if new_score > score_max:
                score_max = new_score
                best = ranking

        return best
