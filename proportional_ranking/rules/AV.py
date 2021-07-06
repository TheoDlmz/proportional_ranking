from proportional_ranking.rules.general import ProportionalRanking
import numpy as np
import math


class AV(ProportionalRanking):
    """
    Approval voting rule:
    The candidates are ordered with respect to their approval scores

    Attributes
    ----------
    profile: np.ndarray
        The profile of voters used for the election

    Examples
    --------
    >>> election = AV()
    >>> election.set_profile([[1, 0, 1, 0, 0], [0, 1, 1, 1, 1], [0, 1, 1, 0, 0]])
    <proportional_ranking.rules.AV.AV object at ...>
    >>> election.print_ranking()
    c > b > a > d > e
    >>> election.representation(0.6, 4)
    13
    """

    def __init__(self):
        super().__init__("Approval voting")

    def ranking(self):
        return np.argsort(-self.profile.sum(axis=0))

    def representation(self, alpha, lambd):
        if alpha <= 0.5:
            raise ValueError("No bound for alpha <= 0.5")
        return math.ceil(lambd * alpha / (2 * alpha - 1))

