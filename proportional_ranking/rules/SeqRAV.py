from proportional_ranking.rules.general import ProportionalRanking
import numpy as np
import math


class SeqRAV(ProportionalRanking):
    """
    SeqRAV Voting Rule:

    RAV stands for reweighted approval voting. Every time we pick a new candidate, the weight of
    satsified voters decrease.

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
    >>> election = SeqRAV([1, 1/2, 1/4, 1/8, 1/16])
    >>> election.set_profile([[1, 1, 1, 0, 0]]*5 + [[0, 0, 1, 1, 1]]*3)
    <proportional_ranking.rules.SeqRAV.SeqRAV object at ...>
    >>> election.print_ranking()
    c > a > d > b > e
    """

    def __init__(self, weights_vector, name="seqRAV"):
        super().__init__(name)
        self.weights_vector = weights_vector

    def ranking(self):
        n, m = self.profile.shape

        scores = self.profile.copy()
        weights = np.zeros(n, dtype=int)
        ranking = []
        for _ in range(m):
            curr_weights = np.array([self.weights_vector[w] for w in weights])
            best_candidate = np.argmax(curr_weights.dot(scores))
            ranking.append(best_candidate)
            for i in range(n):
                if scores[i, best_candidate]:
                    weights[i] += 1
            scores[:, best_candidate] = False

        return ranking


class SeqPAV(SeqRAV):
    """
    SeqPAV voting rule:

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
    alpha: float
        Can be positive or negative.
        alpha = 0 corresponds to classic seqPAV.
        If alpha > 0, it favors bigger groups.
        If alpha < 0, it favors smaller groups

    Examples
    --------
    >>> election = SeqPAV()
    >>> election.set_profile([[1, 1, 1, 0, 0]]*5 + [[0, 0, 1, 1, 1]]*3)
    <proportional_ranking.rules.SeqRAV.SeqPAV object at ...>
    >>> election.print_ranking()
    c > a > b > d > e
    >>> election.representation(0.6, 4)
    139
    """

    def __init__(self, alpha=0):
        name = "seqPAV"
        if alpha != 0:
            name = "seqPAV (alpha = %.2f)" % alpha
        super().__init__(None, name)
        self.alpha = alpha

    def set_profile(self, profile):
        self.profile = np.array(profile)
        self.delete_cache()
        _, m = self.profile.shape
        self.weights_vector = np.array([1/(i + self.alpha) for i in range(1, m+1)])
        return self

    def representation(self, alpha, lambd):
        if self.alpha == 0:
            return math.ceil(2 * (lambd + 1) ** 2 / alpha ** 2)
        else:
            raise NotImplementedError


class GeometricPAV(SeqRAV):
    """
    Geometric PAV:

    Another RAV method in which the weights are (1/p, 1/p², ...)

    Parameters
    ----------
    name: str
        The name of the rule. By default, "seqRAV".
    p: float
        The parameter of the geometric rule

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
    p: float
        The parameter of the geometric rule.

    Examples
    --------
    >>> election = GeometricPAV()
    >>> election.set_profile([[1, 1, 1, 0, 0]]*5 + [[0, 0, 1, 1, 1]]*3)
    <proportional_ranking.rules.SeqRAV.GeometricPAV object at ...>
    >>> election.print_ranking()
    c > a > b > d > e
    """

    def __init__(self, p=1):
        name = "Geometric (%.2f)" % p
        super().__init__(None, name)
        self.p = p

    def set_profile(self, profile):
        self.profile = np.array(profile)
        self.delete_cache()
        _, m = self.profile.shape
        self.weights_vector = np.array([1/self.p**i for i in range(1, m+1)])
        return self
