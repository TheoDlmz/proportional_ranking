def print_profile(profile):
    n, m = profile.shape
    for i in range(n):
        letters = []
        for j in range(m):
            if profile[i, j]:
                letters.append(chr(97+j))
        str_letters = " ".join(letters)
        print("%i : %s" % (i, str_letters))


def print_ranking(ranking):
    letters = [chr(97+x) for x in ranking]
    print(" > ".join(letters))
