from proportional_ranking.rules.general import ProportionalRanking
from itertools import permutations


class scorePAV(ProportionalRanking):
    """
    The rule is defined by Jérôme Lang in his Open Review.
    """
    def __init__(self, scoring_vector=None):
        super().__init__("scorePAV")
        self.scoring_vector = scoring_vector

    def _get_score(self, profile, ranking, vector):
        m = len(ranking)
        s_tot = 0
        for approval in profile:
            s = 0
            div = 1
            for i in range(m):
                if approval[ranking[i]]:
                    s += vector[i] / div
                    div += 1
            s_tot += s
        return s_tot

    def ranking(self):
        profile = self.profile
        n, m = profile.shape

        scoring_vector = self.scoring_vector
        if self.scoring_vector is None:
            scoring_vector = [(m-1-i) for i in range(m)]

        score_max = 0
        best = None
        for ranking in list(permutations(range(m))):
            new_score = self._get_score(profile, ranking, scoring_vector)
            if new_score > score_max:
                score_max = new_score
                best = ranking

        return best
