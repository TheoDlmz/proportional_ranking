import numpy as np
import math
from itertools import combinations


def avg_satisfaction(profile, voters, committee):
    """
    This function compute the average number of approved candidates per voter
    for a subset of voters and a committee of candidates.
    Parameters
    ----------
    profile : np.ndarray
        The approval profile
    voters : np.ndarray
        The subset of voters
    committee : np.ndarray
        The committee of candidates

    Returns
    -------
    float
        The average number of approved candidates in the committee for the subset of voters

    """
    n, m = profile.shape
    valid = np.zeros(m)
    for x in committee:
        valid[x] = 1
    return valid.dot(profile[voters].T).sum() / len(voters)


def justified_demand(profile, voters, k):
    """
    This function compute the justified demand of a subset of voters of an approval profile
    for the first k position of the ranking

    Parameters
    ----------
    profile : np.ndarray
        The approval profile
    voters : np.ndarray
        The subset of voters
    k : int
        The size of the subranking

    Returns
    -------
    int
        The justified demand
    bool
        True if we can still add new candidates for this subset of voters for greater values of k

    """
    n, m = profile.shape
    proportion = int(len(voters) * k / n)
    max_consensus = profile[voters].prod(axis=0).sum()

    return min(proportion, max_consensus), proportion <= max_consensus


def justify(profile, ranking):
    """
    This function compute if a ranking is justified ranking.

    Parameters
    ----------
    profile : np.ndarray
        The approval profile of voters
    ranking : int list
        The ranking of candidates

    Returns
    -------
    bool
        True if and only if the ranking is justified.

    """
    n, m = profile.shape
    min_v = np.infty
    for k in range(1, m + 1):
        j = 1
        prop = math.ceil(n / k)
        while prop <= n:
            list_subset = list(combinations(np.arange(n), prop))
            for subset in list_subset:
                jd, needed = justified_demand(profile, list(subset), k)
                if jd > 0 and needed:
                    af = avg_satisfaction(profile, list(subset), ranking[:k])
                    x = af / jd
                    if x < min_v:
                        min_v = x
                        if min_v < 1:
                            return False
            j += 1
            prop = math.ceil(j * n / k)
    return True


def quality(profile, ranking):
    """
    This function compute the quality of a ranking for a given profile.

    Parameters
    ----------
    profile : np.ndarray
        The approval profile of voters
    ranking : int list
        The ranking of candidates

    Returns
    -------
    bool
        True if and only if the ranking is justified.

    """
    n, m = profile.shape
    min_v = np.infty
    for k in range(1, m + 1):
        prop = math.ceil(n / k)
        for j in range(prop, n + 1):
            list_subset = list(combinations(np.arange(n), j))
            for subset in list_subset:
                jd, needed = justified_demand(profile, list(subset), k)
                if jd > 0 and needed:
                    af = avg_satisfaction(profile, list(subset), ranking[:k])
                    x = af / jd
                    if x < min_v:
                        min_v = x
    return min_v
