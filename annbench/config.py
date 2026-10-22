"""Constants shared by all parts of the assignment."""

K = 10                 # recall@10 everywhere
GT_K = 100             # neighbours stored per query in the data files; we also compute this many
N_TUNE = 5000          # queries[:5000] are for tuning, queries[5000:] for reported numbers
REPEATS = 3            # every timing is repeated this many times and the median is reported

# `--quick`: first 100,000 vectors, and only the first QUICK_QUERIES queries of each half.
QUICK = {"n_base": 100_000, "n_queries_per_half": 1000}
FULL = {"n_base": None, "n_queries_per_half": N_TUNE}
