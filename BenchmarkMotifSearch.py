from BFMotifSearch import BFMotifSearch
from ProbMotifSearch import ProbMotifSearch
from MLMotifSearch import MLMotifSearch

ms = MLMotifSearch(alignment)

profile = ms.get_profile()
ms.get_consensus(profile)
