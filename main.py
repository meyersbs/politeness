#!/usr/bin/env python3

import argparse

def _train(args):
    classifier = Classifier()
    classifier.train(args.documents, args.ntesting)

def _predict(args):
    classifier = Classifier(verbose=True)
    classifier.predict(args.documents)

def _download(args):
    from politeness.data.download import download
    download()

def _set_corenlp_url(args):
    from politeness import helpers
    if args.list:
        print(helpers.get_corenlp_url())
    else:
        helpers.set_corenlp_url(args.url)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Command line access to the Stanford Politeness API. '
        )
    subparsers = parser.add_subparsers(title='Commands', dest='command')
    subparsers.required = True

    parser_train = subparsers.add_parser(
            'train', help='Train politeness classifier.'
        )
    parser_train.add_argument(
            '-d', '--documents', choices=['all', 'wikipedia', 'stackexchange'],
            default='all',
            help='Specify whether to train the classifier on the wikipedia '
                 'documents, the stack-exchange documents, or all documents. '
                 'Default is all.'
        )
    parser_train.add_argument(
            '-n', '--ntesting', default=500,
            help='The number of documents to withhold for testing.'
        )
    parser_train.set_defaults(handler=_train)

    parser_predict = subparsers.add_parser(
            'predict', help='Predict politeness of a sentence.'
        )
    parser_predict.add_argument(
            '-f', '--documents',
            help='The filepath containing the documents to be classified.'
        )
    parser_predict.set_defaults(handler=_predict)

    parser_download = subparsers.add_parser(
            'download', help='Download training data.'
        )
    parser_download.set_defaults(handler=_download)

    parser_url = subparsers.add_parser(
            'url', help='Set the URL for the Stanford CoreNLP Server.'
        )
    parser_url.add_argument(
            '-u', '--url', help='The URL for a CoreNLP Server.'
        )
    parser_url.add_argument(
            '-l', '--list', action='store_true',
            help='Show the currently saved URL.'
        )
    parser_url.set_defaults(handler=_set_corenlp_url)
    args = parser.parse_args()

    from politeness.classifier import Classifier
    args.handler(args)
