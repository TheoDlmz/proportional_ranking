from proportional_ranking.rules.general import ProportionalRanking
from proportional_ranking.utils.quality import quality, justify
from proportional_ranking.utils.printing import print_ranking
from proportional_ranking.rules.AV import AV
from itertools import permutations


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class RankingNotFoundError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class JustifiedRanking(ProportionalRanking):
    """
    This rule picks the first ranking it founds with quality >= 1, i.e. every justified demand
    is fulfilled.

    If no such ranking exists, it raises an error.


    Attributes
    ----------
    profile: np.ndarray
        The profile of voters used for the election

    Examples
    --------
    >>> election = JustifiedRanking()
    >>> election.set_profile([[1, 0, 1, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0]])
    <proportional_ranking.rules.justify.JustifiedRanking object at ...>
    >>> election.print_ranking()
    c > a > b > d > e

    """
    def __init__(self):
        super().__init__("JustifiedRanking")

    def ranking(self):
        profile = self.profile
        n, m = profile.shape
        for ranking in list(permutations(range(m))):
            if justify(profile, ranking):
                return ranking

        raise RankingNotFoundError("No ranking satisfy justify demand")


class MaximizeQuality(ProportionalRanking):
    """
    This rule picks the ranking with maximum quality, as defined in the paper
    Proportional Rankings by Skowron et Al.

    Attributes
    ----------
    profile: np.ndarray
        The profile of voters used for the election

    Examples
    --------
    >>> election = MaximizeQuality()
    >>> election.set_profile([[1, 0, 1, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0]])
    <proportional_ranking.rules.justify.MaximizeQuality object at ...>
    >>> election.print_ranking()
    c > a > b > d > e

    """
    def __init__(self):
        super().__init__("MaxQuality")

    def ranking(self):
        n, m = self.profile.shape
        max_q = 0
        best_ranking = None

        for ranking in list(permutations(range(m))):
            q = quality(self.profile, ranking)
            if q > max_q:
                best_ranking = ranking
                max_q = q

        return best_ranking
