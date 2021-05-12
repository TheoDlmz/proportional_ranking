from proportional_ranking.rules.general import propRanking
from proportional_ranking.utils.quality import quality, justify
from proportional_ranking.utils.printing import print_ranking
from proportional_ranking.rules.AV import AV
from itertools import permutations


class justifyIt(propRanking):
    def __init__(self):
        super().__init__()
        self.name = "JustifyIt"

    def ranking(self):
        profile = self.profile
        n, m = profile.shape
        for ranking in list(permutations(range(m))):
            if justify(profile, ranking):
                return ranking

        return AV().set_profile(profile).ranking()


class maxQuality(propRanking):
    def __init__(self, verbose=False):
        super().__init__()
        self.name = "MaxQuality"
        self.verbose = verbose

    def ranking(self):
        profile = self.profile
        n, m = profile.shape
        max_q = 0
        best_ranking = None

        for ranking in list(permutations(range(m))):
            q = quality(profile, ranking)
            if q > max_q:
                best_ranking = ranking
                max_q = q
            if q >= 1 and self.verbose:
                print_ranking(ranking)

        return best_ranking
