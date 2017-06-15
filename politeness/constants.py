import os

from pkg_resources import resource_filename

TEST_DOCUMENT_PATH = resource_filename(
	'politeness', 'tests/test_documents.txt'
)

TEST_DOCUMENT_JSON_PATH = resource_filename(
	'politeness', 'tests/test_documents.json'
)

POLITENESS_CLASSIFIER_PATH = resource_filename(
	'politeness', 'models/politeness-svm.p'
)

CORENLP_SERVER_URL = resource_filename(
	'politeness', 'corenlp-url.txt'
)

BIGRAM_FEATURES_PATH = resource_filename(
	'politeness', 'features/bigram-feats.p'
)
UNIGRAM_FEATURES_PATH = resource_filename(
	'politeness', 'features/unigram-feats.p'
)

NEGATIVE_WORDS_PATH = resource_filename(
	'politeness', 'features/liu-negative-words.txt'
)
POSITIVE_WORDS_PATH = resource_filename(
	'politeness', 'features/liu-positive-words.txt'
)

PARSED_STACK_EXCHANGE_PATH = resource_filename(
	'politeness', 'data/stack-exchange.parsed.json'
)

PARSED_WIKIPEDIA_PATH = resource_filename(
	'politeness', 'data/wikipedia.parsed.json'
)
