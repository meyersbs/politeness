import unittest

#### PACKAGE IMPORTS ###########################################################
from politeness.constants import CORENLP_SERVER_URL, TEST_DOCUMENT_PATH
from politeness import helpers


class HelpersTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_format_doc(self):
        helpers.set_corenlp_url('artifacts.gccis.rit.edu:41194/')

        expected = [{'sentences': ['I am the walrus.'],
                     'parses': ['ROOT(root-0, walrus-4)',
                     'nsubj(walrus-4, I-1)', 'cop(walrus-4, am-2)',
                     'det(walrus-4, the-3)', 'punct(walrus-4, .-5)']}]
        actual = helpers.format_doc('I am the walrus.')
        self.assertEqual(expected, actual)

    def test_format_doc_str(self):
        helpers.set_corenlp_url('artifacts.gccis.rit.edu:41194/')
        expected = [
            [{'sentences': ['Have you found the answer for your question?'],
              'parses': ['ROOT(root-0, found-3)', 'aux(found-3, Have-1)',
              'nsubj(found-3, you-2)', 'det(answer-5, the-4)',
              'dobj(found-3, answer-5)', 'case(question-8, for-6)',
              'nmod:poss(question-8, your-7)', 'nmod:for(answer-5, question-8)',
              'punct(found-3, ?-9)']},
             {'sentences': ['If yes would you please share it?'],
              'parses': ['ROOT(root-0, please-5)', 'dep(please-5, If-1)',
              'dep(please-5, yes-2)', 'aux(please-5, would-3)',
              'nsubj(please-5, you-4)', 'dobj(please-5, share-6)',
              'dep(please-5, it-7)', 'punct(please-5, ?-8)']}
            ],
            [{'sentences': ['Sorry :) I dont want to hack the system!!'],
              'parses': ['ROOT(root-0, :-RRB--2)', 'amod(:-rrb--2, Sorry-1)',
              'nsubj(want-5, I-3)', 'nsubj:xsubj(hack-7, I-3)',
              'aux(want-5, dont-4)', 'acl:relcl(:-rrb--2, want-5)',
              'mark(hack-7, to-6)', 'xcomp(want-5, hack-7)',
              'det(system-9, the-8)', 'dobj(hack-7, system-9)',
              'nummod(:-rrb--2, !!-10)']},
             {'sentences': [':) is there another way?'],
              'parses': ['ROOT(root-0, :-RRB--1)', 'acl(:-rrb--1, is-2)',
              'expl(is-2, there-3)', 'det(way-5, another-4)',
              'nsubj(is-2, way-5)', 'punct(:-rrb--1, ?-6)']}
            ],
            [{'sentences': ['What are you trying to do?'],
              'parses': ['ROOT(root-0, trying-4)', 'dep(trying-4, What-1)',
              'aux(trying-4, are-2)', 'nsubj(trying-4, you-3)',
              'nsubj:xsubj(do-6, you-3)', 'mark(do-6, to-5)',
              'xcomp(trying-4, do-6)', 'punct(trying-4, ?-7)']
             },
             {'sentences': ['Why can\'t you just store the "Range"?'],
              'parses': ['ROOT(root-0, store-6)', 'advmod(store-6, Why-1)',
              'aux(store-6, ca-2)', "neg(store-6, n't-3)",
              'nsubj(store-6, you-4)', 'advmod(store-6, just-5)',
              'det(range-9, the-7)', 'punct(range-9, ``-8)',
              'dobj(store-6, Range-9)', "punct(range-9, ''-10)",
              'punct(store-6, ?-11)']}
            ],
            [{'sentences': ['This was supposed to have been moved to <url> per '
                            'the cfd.'],
              'parses': ['ROOT(root-0, supposed-3)',
              'nsubjpass(supposed-3, This-1)',
              'nsubjpass:xsubj(moved-7, This-1)',
              'auxpass(supposed-3, was-2)', 'mark(moved-7, to-4)',
              'aux(moved-7, have-5)', 'auxpass(moved-7, been-6)',
              'xcomp(supposed-3, moved-7)', 'mark(<url>-9, to-8)',
              'xcomp(moved-7, <url>-9)', 'case(cfd-12, per-10)',
              'det(cfd-12, the-11)', 'nmod:per(<url>-9, cfd-12)',
              'punct(supposed-3, .-13)']
             },
             {'sentences': ["why wasn't it moved?"],
              'parses': ['ROOT(root-0, moved-5)', 'advmod(moved-5, why-1)',
              'auxpass(moved-5, was-2)', "neg(moved-5, n't-3)",
              'nsubjpass(moved-5, it-4)', 'punct(moved-5, ?-6)']
             }
            ]
        ]

        data = []
        with open(TEST_DOCUMENT_PATH, 'r') as f:
            for line in f.readlines():
                data.append(line.strip('\n'))

        for i, line in enumerate(data):
            actual = helpers.format_doc(line)
            self.assertEqual(expected[i], actual)

    def test_format_doc_dict(self):
        helpers.set_corenlp_url('artifacts.gccis.rit.edu:41194/')

        data = {'If yes would you please share it?': ['ROOT(root-0, please-5)',
                'dep(please-5, If-1)', 'dep(please-5, yes-2)',
                'aux(please-5, would-3)', 'nsubj(please-5, you-4)',
                'dobj(please-5, share-6)', 'dep(please-5, it-7)',
                'punct(please-5, ?-8)']}
        expected = {'sentences': ['If yes would you please share it?'],
                    'parses': ['ROOT(root-0, please-5)', 'dep(please-5, If-1)',
                    'dep(please-5, yes-2)', 'aux(please-5, would-3)',
                    'nsubj(please-5, you-4)', 'dobj(please-5, share-6)',
                    'dep(please-5, it-7)', 'punct(please-5, ?-8)']}
        for sent, deps in data.items():
            actual = helpers.format_doc(sent, deps)
            self.assertEqual(expected, actual)

    def test_url_default(self):
        expected = 'http://0.0.0.0:0000'
        helpers.set_corenlp_url('http://0.0.0.0:0000')
        with open(CORENLP_SERVER_URL, 'r') as f:
            actual = f.read().strip('\n')

        self.assertEqual(expected, actual)

    def test_set_url(self):
        expected = 'http://some-url.org:1234'
        helpers.set_corenlp_url('some-url.org:1234')
        with open(CORENLP_SERVER_URL, 'r') as f:
            actual = f.read().strip('\n')

        self.assertEqual(expected, actual)
