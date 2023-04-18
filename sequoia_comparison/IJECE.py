from dataset.utils import set_path
from sequoia_comparison.utils import get_single_scores
from scores.get_scores import print_avg_scores

set_path()
print_avg_scores(get_single_scores, True)
