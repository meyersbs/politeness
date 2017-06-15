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
                'Sorry : I dont want to hack the system!!':
                    [0.5, 0.5]
            },
            {
                ': is there another way?':
                    [0.47036089792593772, 0.52963910207406228]
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
            {'document': [0.36745111795347302, 0.63254888204652693]}
        ]

        actual = self.classifier.predict(data)
        self.assertEqual(expected, actual)
