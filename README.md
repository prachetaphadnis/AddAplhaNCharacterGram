# Simple n-gram character level model with add alpha smoothing

The model optimises for alpha using grid search. Grid values can be chosen by setting alpha_options to appropriate values.

usage:
python add_alpha_ngram.py training_file [training_file2 ...] test_file

Code writes the smooth model to disk on completion and outputs perplexities on test file for eacch training set.
