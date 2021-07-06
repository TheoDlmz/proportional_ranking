from proportional_ranking.rules.general import ProportionalRanking
from itertools import permutations

import numpy as np

class seqScorePAV(ProportionalRanking):
    """
    Sequential version of Jeromes new approval to ranking rule.
    For the score vector (1,1,1,...) this rule is equivalent to
    sequential PAV.

    Ties are broken in alphabetically.

    If length(scorevector) < m, then we add 0's to the scorevector.
    If scorevector is only an int x, we set scorevector to (x,x,x,...).
    If scorevector euqals 'b', we set scorevector to (m-1, m-2, ...).
    """

    def __init__(self, scorevec=1):
        super().__init__("seqScorePAV with " + str(scorevec))
        self.scorevec = scorevec

    def set_scorevector(self, scorevec):
        self.name = "seqScorePAV with " + str(scorevec)
        self.scorevec = scorevec

    def __adjust_scorevector(self, num_cands):
        """
        If length(scorevector) < m, then we add 0's to the scorevector.
        If scorevector is only an int x, we set scorevector to (x,x,x,...).
        If scorevector euqals 'b', we set scorevector to (m-1, m-2, ...).
        """

        if self.scorevec == 'b':
            scorevec = np.arange(num_cands)[::-1]
        elif isinstance(self.scorevec, int):
            scorevec = np.ones(num_cands) * self.scorevec
        else:
            scorevec = self.scorevec
        if len(scorevec) < num_cands:
            scorevec += [0] * (num_cands - len(scorevec))
        return scorevec

    def __individual_utility(self, voter, ranking, scorevec):
        """ Compute utility a voter obtains from a ranking. """

        utils = 0
        k = 0
        for rank, cand in enumerate(ranking):
            if self.profile[voter][cand]:
                k += 1
                utils += scorevec[rank] / k
        return utils

    def _overall_utility(self, num_voters, ranking, scorevec):
        """ Compute utility all voters obtain from a ranking. """

        utils = 0
        for voter in range(num_voters):
            utils += self.__individual_utility(voter, ranking, scorevec)
        return utils

    def ranking(self):
        """ Compute ranking w.r.t. self.scorevec. """

        profile = self.profile
        n, m = profile.shape
        scorevec = self.__adjust_scorevector(m)

        # construct ranking
        ranking = []
        for _ in range(m):
            next_cand = -1
            best_gain = -1
            without_cand = self._overall_utility(n, ranking, scorevec)
            for cand in range(m):
                if cand in ranking:
                    continue
                # compute gain in utility this candidate invokes
                with_cand = self._overall_utility(n, ranking + [cand], scorevec)
                if with_cand - without_cand > best_gain:
                    next_cand = cand
                    best_gain = with_cand - without_cand
            # append cand with best gain to ranking
            ranking.append(next_cand)

        return ranking

