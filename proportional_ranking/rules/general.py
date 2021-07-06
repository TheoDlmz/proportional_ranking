from proportional_ranking.utils.cache import DeleteCacheMixin, cached_property
from proportional_ranking.utils.quality import quality, justify
from proportional_ranking.utils.printing import print_ranking
import numpy as np


class propRanking(DeleteCacheMixin):
    """
    General class for Proportional Ranking rules.

    Parameters
    ----------
    name: str
        The name of the voting rule

    Attributes
    ----------
    name: str
        The name of the voting rule
    profile: np.ndarray
        A matrix representing an election.
        ``profile[i,j] = 1`` if voter i
        approves candidate j
    """
    def __init__(self, name=""):
        self.profile = None
        self.name = name

    def set_profile(self, profile):
        """
        Update the profile of voters.

        Parameters
        ----------
        profile: np.ndarray or list
            The new profile of voters

        Returns
        -------
        propRanking
            Itself

        """
        self.profile = np.array(profile)
        self.delete_cache()
        return self

    @cached_property
    def ranking(self):
        """
        Compute a ranking of the candidates

        Returns
        -------
        np.ndarray
            ordered list of the candidates.

        """
        raise NotImplementedError

    def print_ranking(self):
        """
        Print nicely the ranking

        Returns
        -------
        None

        """
        print_ranking(self.ranking())

    @cached_property
    def quality(self):
        """
        Compute the quality of the current ranking. A quality > 1 means that
        the ranking is proportional and respects justified demand.

        Returns
        -------
        float
            The quality of the ranking.

        """
        p = self.profile
        r = self.ranking()
        return quality(p, r)

    @cached_property
    def justifiable(self):
        """
        Compute quickly if the ranking respects justified demand, i.e. if the quality is
        > 1

        Returns
        -------
        bool
            If True, the ranking satisfy justified demand

        """
        p = self.profile
        r = self.ranking()
        return justify(p, r)

    def name(self):
        """
        Return the name of the rule

        Returns
        -------
        str
            The name of the rule

        """
        return self.name

    def representation(self, alpha, lambd):
        """
        Return the upper bound for k(alpha,lambda) group representation. See the paper
        Proportional Ranking (Skowron et Al) for more information.

        Parameters
        ----------
        alpha: float
            Betweeen 0 and 1, the proportion of voters that approve the same candidates.

        lambd: int
            The number of candidates commonly approved by the group of voter.

        Returns
        -------
        int
            The upper bound on the number of candidates in the ranking before all lambda cadidates
            already appeared.

        """
        raise NotImplementedError
