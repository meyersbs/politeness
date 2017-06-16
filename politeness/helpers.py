import _pickle
import json
import os
import re
import requests
import sys
import traceback

from nltk import sent_tokenize

#### PACKAGE IMPORTS ###########################################################
from politeness.constants import (CORENLP_SERVER_URL, PARSED_WIKIPEDIA_PATH,
                                  PARSED_STACK_EXCHANGE_PATH)
from politeness.strategies import (check_elems_for_strategy, initial_polar,
                                   aux_polar)

#### GLOBALS ###################################################################
HEADERS = {'Content-Type': 'application/x-www-form-urlencoded; charset=utf-8'}
PARAMS = {'properties': "{'annotators': 'tokenize,ssplit,pos,parse,depparse'}"}
with open(CORENLP_SERVER_URL, 'r') as f:
    URL = f.read().strip('\n')


#### REQUEST UTILITIES #########################################################
def check_is_request(document):
    """ Heuristic to determine whether a document looks like a request. """
    for sentence, parse in zip(document['sentences'], document['parses']):
        if "?" in sentence:
            return True
        if (check_elems_for_strategy(parse, initial_polar) or
                check_elems_for_strategy(parse, aux_polar)):
            return True
    return False


#### DOCUMENT FORMATTING UTILITIES #############################################
def __format_doc_str(doc_text):
    raw_parses, results = [], []
    sents = get_sentences(doc_text)
    for sent in sents:
        raw_parses.append(get_parses(sent))

    for raw in raw_parses:
        result = {'parses': [], 'sentences': []}
        for dep in raw['deps']:
            result['parses'].append(clean_depparse(dep))
        result['sentences'].append(raw['sent'])
        results.append(result)

    return results


def __format_doc_dict(doc_text, doc_parses):
    raw_parses, results = [], []
    sents = get_sentences(doc_text)
    for sent in sents:
        raw_parses.append(get_parses(sent, doc_parses))

    for not_raw in raw_parses:
        result = {'parses': [], 'sentences': []}
        for dep in not_raw['deps']:
            result['parses'].append(dep)
        result['sentences'].append(not_raw['sent'])
        results = result

    return results


def format_doc(doc_text, doc_parses=None):
    """
    Given some doc_text (str), process the text and convert it into the expected
    format for prediction.
    """
    results, sents, raw_parses = [], [], []
    if doc_parses is None:
        results = __format_doc_str(doc_text)
    else:
        results = __format_doc_dict(doc_text, doc_parses) #[0]

    return results


def get_sentences(doc_text):
    """
    Given some doc_text (str), break the text into a list of sentences
    using NLTK's sent_tokenize function.
    """
    temp = doc_text.strip().split('\n')
    sents = []
    for s in temp:
        sents += sent_tokenize(s.strip())

    return sents


def clean_depparse(dep):
    """
    Given a dependency dictionary, return a formatted string representation.
    """
    return str(dep['dep'] + "(" + dep['governorGloss'].lower() + "-" +
               str(dep['governor']) + ", " + dep['dependentGloss'] + "-" +
               str(dep['dependent']) + ")")


def get_parses(sent, parses=None):
    """
    Given a sentence, send the sentence to a Stanford CoreNLP server for
    processing.
    """
    global HEADERS, PARAMS, URL

    parse = {'deps': [], 'sent': ""}
    if parses is not None:
        parse['deps'] = parses
        parse['sent'] = sent

        return parse
    else:
        if URL == "http://0.0.0.0:0000/":
            sys.stderr.write('ERROR: No url has been provided.\n')
            sys.stderr.write('  Please run the following to set one:\n')
            sys.stderr.write("  set_corenlp_url('http://some-url.org:1234')\n")
            sys.exit()

        try:
            response = requests.post(
                    URL, params=PARAMS, headers=HEADERS,
                    data=sent.encode('UTF-8')
                )
            response.raise_for_status()
            for sentence in response.json()['sentences']:
                parse['deps'] = sentence['enhancedPlusPlusDependencies']
                parse['sent'] = sent
        except Exception as e:
            sys.stderr.write('Exception\n')
            sys.stderr.write('  Sentence: {}\n'.format(sent[:50]))
            extype, exvalue, extrace = sys.exc_info()
            traceback.print_exception(extype, exvalue, extrace)
            parse = {'deps': 'X', 'sent': 'X'}
            return parse

        return parse


#### GENERAL UTILITY FUNCTIONS #################################################
def set_corenlp_url(url):
    """ Set the URL for the Stanford CoreNLP server to use. """
    with open(CORENLP_SERVER_URL, 'w') as urlfile:
        if not re.match(r"(http://|https://)", url):
            url = "http://" + url
        urlfile.write(url)

    refresh_URL()

def get_corenlp_url():
    """ Get the currently saved URL for the Stanford CoreNLP server. """
    with open(CORENLP_SERVER_URL, 'r') as urlfile:
        url = urlfile.read().strip('\n')

    return url


def refresh_URL():
    global URL
    with open(CORENLP_SERVER_URL, 'r') as f:
        for line in f.readlines():
            URL = line.strip('\n')
            break


def dump(obj, filepath):
    """ Serialize the given obj to the given filepath. """
    with open(filepath, 'wb') as file:
        _pickle.dump(obj, file)


def load(filepath):
    """ Deserialize the obj located at filepath. """
    if not os.path.exists(filepath):
        raise FileNotFoundError('No such file: {}'.format(filepath))

    with open(filepath, 'rb') as file:
        return _pickle.load(file)


def load_data(documents):
    """ Load the specified data files. """
    all_docs = []
    if documents == 'all':
        print("Gathering All Available Docs...")
        all_docs = json.loads(open(PARSED_STACK_EXCHANGE_PATH, 'r').read()) + \
            json.loads(open(PARSED_WIKIPEDIA_PATH, 'r').read())
    elif documents == 'wikipedia':
        print("Gathering All Wikipedia Docs...")
        all_docs = json.loads(open(PARSED_WIKIPEDIA_PATH, 'r').read())
    elif documents == 'stackexchange':
        print("Gathering All Stack Exchange Docs...")
        all_docs = json.loads(open(PARSED_STACK_EXCHANGE_PATH, 'r').read())
    else:
        if os.path.exists(documents):
            all_docs = json.loads(open(documents, 'r').read())
        else:
            print("Defaulting to All Available Docs...")
            all_docs = json.loads(open(PARSED_STACK_EXCHANGE_PATH, 'r').read()) + \
                json.loads(open(PARSED_WIKIPEDIA_PATH, 'r').read())

    return all_docs

def get_elapsed(begin, end):
    """
    Return the number of minutes that passed between the specified beginning
    and end.
    """
    return (end - begin).total_seconds() / 60
