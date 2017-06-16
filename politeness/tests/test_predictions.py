import unittest

#### PACKAGE IMPORTS ###########################################################
from politeness.constants import CORENLP_SERVER_URL, TEST_DOCUMENT_PATH
from politeness.classifier import Classifier
from politeness import helpers


class PredictionsTestCase(unittest.TestCase):
    def setUp(self):
        self.classifier = Classifier()

    def test_predictions(self):
        helpers.set_corenlp_url('artifacts.gccis.rit.edu:41194')

        data = "Have you found the answer for your question? If yes would you" \
               " please share it? Sorry :) I dont want to hack the system!! :" \
               ") is there another way? What are you trying to do?  Why can't" \
               " you just store the \"Range\"? This was supposed to have been" \
               " moved to <url> per the cfd. why wasn't it moved?"
        expected = [
            {
                'Have you found the answer for your question?':
                    [0.45793466358055329, 0.54206533641944665]
            },
            {
                'If yes would you please share it?':
                    [0.47243183562775615, 0.52756816437224363]
            },
            {
                'Sorry :) I dont want to hack the system!!':
                    [0.54823398057393613, 0.45176601942606376]
            },
            {
                ':) is there another way?':
                    [0.54151149263615428, 0.45848850736384572]
            },
            {
                'What are you trying to do?':
                    [0.25075788316047232, 0.74924211683952746]
            },
            {
                'Why can\'t you just store the "Range"?':
                    [0.10255615890730475, 0.89744384109269537]
            },
            {
                'This was supposed to have been moved to <url> per the cfd.':
                    [0.38666486673559936, 0.61333513326440048]
            },
            {
                "why wasn't it moved?":
                    [0.29890263769016051, 0.70109736230983943]
            },
            {'document': [0.38237418986399213, 0.61762581013600781]}
        ]

        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)


        data = {'sentence': 'If yes would you please share it?',
                'parses': ['ROOT(root-0, please-5)', 'dep(please-5, If-1)',
                           'dep(please-5, yes-2)', 'aux(please-5, would-3)',
                           'nsubj(please-5, you-4)', 'dobj(please-5, share-6)',
                           'dep(please-5, it-7)', 'punct(please-5, ?-8)']}
        expected = [
            {
                'If yes would you please share it?':
                    [0.47243183562775615, 0.52756816437224363]
            },
            {'document': [0.47243183562775615, 0.52756816437224363]}
        ]

        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)
