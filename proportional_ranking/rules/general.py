from proportional_ranking.utils.cache import DeleteCacheMixin, cached_property
from proportional_ranking.utils.quality import quality, justify
from proportional_ranking.utils.printing import print_ranking


class propRanking(DeleteCacheMixin):

    def __init__(self):
        self.profile = None
        self.name = ""

    def set_profile(self, p):
        self.profile = p
        self.delete_cache()
        return self

    @cached_property
    def ranking(self):
        raise NotImplementedError

    def print_ranking(self):
        print_ranking(self.ranking())

    @cached_property
    def quality(self):
        p = self.profile
        r = self.ranking()
        return quality(p, r)

    @cached_property
    def justifiable(self):
        p = self.profile
        r = self.ranking()
        return justify(p, r)

    def name(self):
        return self.name

    def representation(self, alpha, lambd):
        raise NotImplementedError
