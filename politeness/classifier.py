import numpy as np
import os
import random

from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.metrics import classification_report

#### PACKAGE IMPORTS ###########################################################
from politeness.constants import POLITENESS_CLASSIFIER_PATH
from politeness import helpers
from politeness.vectorizer import PolitenessFeatureVectorizer


class Classifier(object):
    def __init__(self):
        self.classifier = None
        self.vectorizer = None


    def train(self, documents, ntesting=500):
        if self.vectorizer is None:
            self.vectorizer = PolitenessFeatureVectorizer()

        documents = helpers.load_data(documents)

        # Generate and persist list of unigrams, bigrams
        PolitenessFeatureVectorizer.generate_bow_features(documents)

        # For good luck
        random.shuffle(documents)
        testing = documents[-ntesting:]
        documents = documents[:-ntesting]

        # SAVE FOR NOW
        helpers.dump(testing, "testing-data.p")

        X, y = self._documents2feature_vectors(documents)
        Xtest, ytest = self._documents2feature_vectors(testing)

        self.classifier = svm.SVC(C=0.02, kernel='linear', probability=True)
        self.classifier.fit(X, y)

        # Test
        y_pred = self.classifier.predict(Xtest)
        print(classification_report(ytest, y_pred))

        print("Saving Classifier to Disk...")
        self._dump()


    def predict(self, doc_path):
        if self.classifier is None or self.vectorizer is None:
            self._load()

        doc_text = None
        parsed_docs, polite, impolite = [], [], []
        # {'sentences': [self.text], 'parses': [self.depparses]}
        if type(doc_path) == dict:
            sent = doc_path['sentence']
            deps = doc_path['parses']
            parsed_docs.append(helpers.format_doc(sent, deps))
        elif type(doc_path) == str:
            if os.path.exists(doc_path):
                with open(doc_path, "r") as doc:
                    doc_text = doc.read()
            else:
                doc_text = doc_path

            parsed_docs = helpers.format_doc(doc_text)

        output = []
        for i, doc in enumerate(parsed_docs):
            probs = self._score(doc)
            polite.append(probs['polite'])
            impolite.append(probs['impolite'])
            print("\n====\nSentence " + str(i) + ":\n" + str(doc['sentences'][0]))
            print("\tP(polite) = %.3f" % probs['polite'])
            print("\tP(impolite) = %.3f" % probs['impolite'])

            output.append({str(doc['sentences'][0]): [polite[i], impolite[i]]})

        print("\n====\nDocument:")
        print("\tP(polite) = %.3f" % np.mean(polite))
        print("\tP(impolite) = %.3f" % np.mean(impolite))

        output.append({"document": [np.mean(polite), np.mean(impolite)]})
        return output


    def _score(self, request):
        # Vectorizer returns {feature-name: value} dict
        features = self.vectorizer.features(request)
        fv = [features[f] for f in sorted(features.keys())]
        # Single-row sparse matrix
        X = csr_matrix(np.asarray([fv]))
        probs = self.classifier.predict_proba(X)

        return {"polite": probs[0][1], "impolite": probs[0][0]}


    def _documents2feature_vectors(self, documents):
        """ Generate feature vectors for the given list of documents. """
        print("Calculating Feature Vectors...")
        fks = False
        X, y = [], []
        cnt = 0
        for d in documents:
            fs = self.vectorizer.features(d)
            if not fks:
                fks = sorted(fs.keys())
            fv = [fs[f] for f in fks]
            # If politeness score > 0.0, the doc is polite (class = 1)
            try:
                l = 1 if float(d['score']) > 0.0 else 0
            except ValueError:
                l = 0
            X.append(fv)
            y.append(l)
            cnt += 1
        X = csr_matrix(np.asarray(X))
        y = np.asarray(y)
        return X, y

    def _dump(self):
        helpers.dump(self.classifier, POLITENESS_CLASSIFIER_PATH)

    def _load(self):
        self.classifier = helpers.load(POLITENESS_CLASSIFIER_PATH)
        self.vectorizer = PolitenessFeatureVectorizer()
