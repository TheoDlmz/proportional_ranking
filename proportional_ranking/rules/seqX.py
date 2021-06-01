from proportional_ranking.rules.general import propRanking
from itertools import permutations
from fractions import Fraction

import numpy as np

class seqX(propRanking):
    """
    Sequential version of Rule X. By default uses an increment of 1/n.
    """

    def __init__(self, increment=0):
        super().__init__()
        if increment <= 0: 
            self.incr = 0
        else:
            self.incr = increment
        self.name = "seqX with " + str(self.incr)

    def set_increment(self, increment):
        if increment <= 0: 
            self.incr = 0
        else:
            self.incr = increment
        self.name = "seqScorePAV with " + str(self.incr)

    def __get_min_q(self, budgets, cand):
        """ 
        Compute minimal q for Rule X and seqX.
        Shamelessly copied and adpated from Martin Lackners code.
        """

        profile = self.profile

        rich = set([v for v, pref in enumerate(profile)
                    if pref[cand]])
        poor = set()

        while len(rich) > 0:
            poor_budget = sum(budgets[v] for v in poor)
            q = Fraction(1 - poor_budget, len(rich))
            new_poor = set([v for v in rich
                            if budgets[v] < q])
            if len(new_poor) == 0:
                return q
            rich -= new_poor
            poor.update(new_poor)

        return None  # not sufficient budget available

    def ranking(self):
        """ Compute ranking w.r.t. self.incr. """

        profile = self.profile
        n, m = profile.shape

        budgets = {v: 0 for v in range(n)}
        candidates = range(m)
        ranking = []
        if self.incr == 0:
            # use 1/n as budget increase
            budget_increase = Fraction(1, n)
        else:
            budget_increase = self.incr

        while len(ranking) < m:
            # setting up new budgets
            for voter, budget in budgets.items():
                budgets[voter] = budget + budget_increase

            # reimplement RuleX to get the committee, the remaining budget
            enough_budget = 1
            while enough_budget:
                curr_cands = set(candidates) - set(ranking)
                min_q = {}
                for c in curr_cands:
                    q = self.__get_min_q(budgets, c)
                    if q is not None:
                        min_q[c] = q
                if len(min_q) > 0:
                    # i.e., one or more candidates are affordable
                    next_cands = [c for c in min_q.keys()
                                if min_q[c] == min(min_q.values())]
                    for next_cand in next_cands:
                        new_budgets = dict(budgets)
                        for v, pref in enumerate(profile):
                            if next_cand in pref:
                                new_budgets[v] -= min(budgets[v], min_q[next_cand])
                        ranking += [next_cand]
                        budgets = new_budgets
                        break
                else:   # no candidate is affordable or committee is full
                    enough_budget = 0
                    break
        return ranking
