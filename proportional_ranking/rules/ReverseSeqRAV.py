from proportional_ranking.rules.general import ProportionalRanking
import numpy as np


class ReverseSeqRAV(ProportionalRanking):
    """
    reverseSeqRAV Voting Rule:

    RAV stands for reweighted approval voting, but the ranking is built from the end. The candidate with the
    lowest marginal contribution is added at the very end, then the second one with least marginal contribution
    and updated weights.

    Parameters
    ----------
    name: str
        The name of the rule. By default, "seqRAV".
    weights_vector: float list
        The vector of weights. If a voter already have k approved candidates in the ranking, its weight
        is of ``weights_vector[k]``

    Attributes
    ----------
    name: str
        The name of the voting rule
    profile: np.ndarray
        A matrix representing an election.
        ``profile[i,j] = 1`` if voter i
        approves candidate j
    weights_vector: float list
        The vector of weights. If a voter already have k approved candidates in the ranking, its weight
        is of ``weights_vector[k]``

    Examples
    --------
    >>> election = ReverseSeqRAV([1, 1/2, 1/4, 1/8, 1/16])
    >>> election.set_profile([[1, 1, 1, 0, 0]]*5 + [[0, 0, 1, 1, 1]]*3)
    <proportional_ranking.rules.ReverseSeqRAV.ReverseSeqRAV object at ...>
    >>> election.print_ranking()
    c > a > d > b > e
    """

    def __init__(self, weights_vector, name="reverseSeqRAV"):
        super().__init__(name)
        self.weights_vector = weights_vector

    def ranking(self):
        n, m = self.profile.shape

        scores = self.profile.copy()
        weights = scores.sum(axis=1)
        ranking = []
        weights_vector = [0] + self.weights_vector

        for _ in range(m):
            curr_weights = np.array([weights_vector[x] for x in weights])
            s = np.argsort(curr_weights.dot(scores)[::-1])
            s = len(s) - 1 - s
            worst_candidate = -1
            for candidate in s:
                if candidate not in ranking:
                    worst_candidate = candidate
                    break

            ranking.append(worst_candidate)

            for k in range(n):
                if scores[k, worst_candidate]:
                    weights[k] -= 1

            scores[:, worst_candidate] = False

        return ranking[::-1]


class ReverseSeqPAV(ReverseSeqRAV):
    """
    reverseSeqPAV voting rule:

    PAV stands for Proportional Approval Voting. The weights vector is (1, 1/2, 1/3, ...)

    `alpha` parameter change the weights to favors big or small groups
    (depending on the sign of alpha)

    Parameters
    ----------
    name: str
        The name of the rule. By default, "seqRAV".
    alpha: float
        Can be positive or negative.
        alpha = 0 corresponds to classic seqPAV.
        If alpha > 0, it favors bigger groups.
        If alpha < 0, it favors smaller groups

    Attributes
    ----------
    name: str
        The name of the voting rule
    profile: np.ndarray
        A matrix representing an election.
        ``profile[i,j] = 1`` if voter i
        approves candidate j
    weights_vector: float list
        The vector of weights. If a voter already have k approved candidates in the ranking, its weight
        is of ``weights_vector[k]``

    Examples
    --------
    >>> election = ReverseSeqPAV()
    >>> election.set_profile([[1, 1, 1, 0, 0]]*5 + [[0, 0, 1, 1, 1]]*3)
    <proportional_ranking.rules.ReverseSeqRAV.ReverseSeqPAV object at ...>
    >>> election.print_ranking()
    c > a > b > d > e
    """

    def __init__(self, alpha=0):
        name = "reverseSeqPAV"
        if alpha != 0:
            name = "reverseSeqPAV (alpha = %.2f)" % alpha
        super().__init__(None, name)
        self.alpha = alpha

    def set_profile(self, profile):
        self.profile = np.array(profile)
        self.delete_cache()
        _, m = self.profile.shape
        self.weights_vector = np.array([1/(i + self.alpha) for i in range(1, m+1)])
        return self
