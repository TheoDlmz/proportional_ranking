import numpy as np
from proportional_ranking.experiments.data_generation import generate_profile


def compare_rules(n, m, rules, iterations=100, verbose=False):
    n_rules = len(rules)
    avg = np.zeros(n_rules + 1)
    success = np.zeros(n_rules + 1)
    i = 0
    while i < iterations:
        profile = generate_profile(n, m, proba=np.random.rand())
        if np.min(profile.sum(axis=0)) == 0 or np.min(profile.sum(axis=1)) == 0:
            continue

        if verbose and i % 50 == 0:
            print(i)

        success_i = []
        qualities = []
        ranking = []
        for rule in rules:
            rule.set_profile(profile)
            ranking.append(rule.ranking())
            success_i.append(rule.justifiable)
            qualities.append(rule.quality)

        success_i.append(np.max(success_i))
        qualities.append(np.max(qualities))
        success += success_i
        avg += qualities
        i += 1

    return avg / iterations, success / iterations


def test_profile(profile, rules):
    for r in rules:
        rule = r.set_profile(profile)
        print("%s : %.2f" % (rule.name, rule.quality))
        rule.print_ranking()
