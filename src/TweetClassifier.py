from DataProcessor import DataProcessor

import getopt, sys, logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV

from nltk import *


class TweetClassifier:
    """   : 1 positive opinion, -1: negative opinion,
0: neutral or no opinion, 2: mixed opinion (there is positive and negative
opinion)

Objective: build a classifier to classify tweets into three classes: 1, -1,
0. You can ignore the mixed class.

    """
    def __init__(self):

        self.classifiers = []
        self.classifiers.append('MultinomialNB')
        # self.classifiers.append('BernoulliNB')
        self.classifiers.append('SVM')
        # self.classifiers.append('RandomForest')
        # self.classifiers.append('DecisionTreeClassifier')


    def my_tokenizer(self, text):
        stemmer = PorterStemmer()
        words = [ re.match('^[a-zA-Z\'-]+', w).group() for w in text.split(" ") if re.match('^[a-zA-Z\'-]+', w) != None]
        # return [stemmer.stem(w) for w in words]
        return words

    def get_classifier(self, name):

        vect = CountVectorizer(analyzer="word", strip_accents='unicode', ngram_range=(1, 2), tokenizer=self.my_tokenizer,
                               stop_words=None, lowercase=True) # max_features=35000

        clf_params = [('vect', vect), ('tfidf', TfidfTransformer())]
        if name == 'MultinomialNB':
            clf_params.append(('clf', MultinomialNB()))
        elif name == 'BernoulliNB':
            clf_params.append(('clf', BernoulliNB()))
        elif name == 'SVM':
            clf_params.append(('clf', LinearSVC(loss='hinge')))
        elif name == 'RandomForest':
            clf_params.append(('clf', RandomForestClassifier()))
        elif name == 'DecisionTreeClassifier':
            clf_params.append(('clf', DecisionTreeClassifier()))

        text_clf = Pipeline(clf_params)

        return text_clf

    def grid_search(self, classifier_name, data):

        text_clf = self.get_classifier(classifier_name)

        print text_clf.get_params().keys()

        # parameters = {'clf__alpha':(0.725, 0.75, 1.0)}
        parameters = {'clf__C': (0.1, .2, .4, .6, .8, 1.0)}

        gs_clf = GridSearchCV(estimator=text_clf, param_grid=parameters, cv=10, n_jobs=1)

        gs_clf = gs_clf.fit(data['text'][:5000], data['target'][:5000])

        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))

        print score

    def crossvalidate(self, data, num_fold):

        text = data['text']
        target = data['target']

        assert(len(text) == len(target))

        classifier_report = {}

        for name in self.classifiers:

            kf = KFold(len(text), n_folds=num_fold, shuffle=True, random_state=None) # shuffle=True, random_state=660018239
            avg_accuracy = 0.0

            pos_scores = [0.0, 0.0, 0.0]    #precision, recall, fscore
            neg_scores = [0.0, 0.0, 0.0]

            for train_index, test_index in kf:
                # print train_index, test_index

                train_d = [text[i] for i in train_index]
                test_d = [text[i] for i in test_index]

                train_t = [target[i] for i in train_index]
                test_t = [target[i] for i in test_index]

                clf = self.get_classifier(name)

                # vect = TfidfVectorizer(analyzer="word", strip_accents='unicode', ngram_range=(1, 2))
                # X_train_counts = vect.fit_transform(train_d)
                # print vect.vocabulary_['america'], vect.idf_[ vect.vocabulary_['america'] ]
                # print ''
                #
                # print vect.idf_

                text_fit = clf.fit(train_d, train_t)
                predicted = text_fit.predict(test_d)

                avg_accuracy += accuracy_score(test_t, predicted)

                # precision, recall, fscore, dummy = precision_recall_fscore_support(test_t, predicted.astype(int), average='macro')

                cm = confusion_matrix(test_t, predicted, labels=[-1, 0, 1])

                # print cm

                l = 0.000001
                pos_scores[0] += (float(cm[2][2]) / (cm[2][2] + cm[1][2] + cm[0][2] + l))
                pos_scores[1] += (float(cm[2][2]) / (cm[2][0] + cm[2][1] + cm[2][2] + l))
                pos_scores[2] = (2 * pos_scores[0] * pos_scores[1]) / (pos_scores[0] + pos_scores[1] + l)

                neg_scores[0] += (float(cm[0][0]) / (cm[0][0] + cm[1][0] + cm[2][0]) + l)
                neg_scores[1] += (float(cm[0][0]) / (cm[0][0] + cm[0][1] + cm[0][2]) + l)
                neg_scores[2] = (2 * neg_scores[0] * neg_scores[1]) / (neg_scores[0] + neg_scores[1] + l)

                # print(classification_report(test_t, predicted, labels=[-1, 0, 1], target_names=['negative', 'neutral', 'positive']))


            report = {
                'precision' :   ((pos_scores[0] / num_fold), (neg_scores[0] / num_fold)),
                'recall'    :   ((pos_scores[1] / num_fold), (neg_scores[1] / num_fold)),
                'fscore'    :   ((pos_scores[2] / num_fold), (neg_scores[2] / num_fold)),
                'accuracy'  :   (avg_accuracy / num_fold)
            }

            classifier_report[name] = report

        return classifier_report

    def train_test(self, train_data, test_data):

        classifier_report = {}

        for name in self.classifiers:
            accuracy = 0.0

            pos_scores = [0.0, 0.0, 0.0]    #precision, recall, fscore
            neg_scores = [0.0, 0.0, 0.0]

            clf = self.get_classifier(name)

            text_fit = clf.fit(train_data['text'], train_data['target'])
            predicted = text_fit.predict(test_data['text'])

            accuracy += accuracy_score(test_data['target'], predicted)

            cm = confusion_matrix(test_data['target'], predicted, labels=[-1, 0, 1])

            l = 0.000001
            pos_scores[0] += (float(cm[2][2]) / (cm[2][2] + cm[1][2] + cm[0][2] + l))
            pos_scores[1] += (float(cm[2][2]) / (cm[2][0] + cm[2][1] + cm[2][2] + l))
            pos_scores[2] = (2 * pos_scores[0] * pos_scores[1]) / (pos_scores[0] + pos_scores[1] + l)

            neg_scores[0] += (float(cm[0][0]) / (cm[0][0] + cm[1][0] + cm[2][0]) + l)
            neg_scores[1] += (float(cm[0][0]) / (cm[0][0] + cm[0][1] + cm[0][2]) + l)
            neg_scores[2] = (2 * neg_scores[0] * neg_scores[1]) / (neg_scores[0] + neg_scores[1] + l)

            report = {
                'precision' :   (pos_scores[0], neg_scores[0]),
                'recall'    :   (pos_scores[1], neg_scores[1]),
                'fscore'    :   (pos_scores[2], neg_scores[2]),
                'accuracy'  :   accuracy
            }

            classifier_report[name] = report

        return classifier_report


def main():

    data_path = "../data/training-Obama-Romney-tweets.xlsx"
    test_data_path = ''
    # test_data_path = '../data/testing-Obama-Romney-tweets-3labels.xlsx'

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:t:")
        for o, a in opts:
            if o == '-d':
                data_path = a
            elif o == '-t':
                test_data_path = a

    except getopt.GetoptError as err:
        # print help information and exit:
        print str(err)
        print 'read the readme file to know how to run this project'
        sys.exit(2)


    dp = DataProcessor(data_path)
    tc = TweetClassifier()

    if test_data_path != '':

        dpt = DataProcessor(test_data_path)

        print '\n****** OBAMA ******\n'
        data = dp.load_excel_data('Obama')
        data_test = dpt.load_excel_data('Obama')
        report = tc.train_test(data, data_test)
        DataProcessor.print_report(report)

        print '\n****** ROMNEY ******\n'
        data = dp.load_excel_data('Romney')
        data_test = dpt.load_excel_data('Romney')
        report = tc.train_test(data, data_test)
        DataProcessor.print_report(report)

    else:
        print '\n****** OBAMA ******\n'
        data = dp.load_excel_data('Obama')
        report = tc.crossvalidate(data, 10)
        DataProcessor.print_report(report)

        print '\n****** ROMNEY ******\n'
        data = dp.load_excel_data('Romney')
        report = tc.crossvalidate(data, 10)
        DataProcessor.print_report(report)

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    main()




# vect = CountVectorizer(analyzer="word", strip_accents='unicode', ngram_range=(1, 1), tokenizer=tc.my_tokenizer,
        #                        stop_words='english', lowercase=True)
        # vect.fit(data['text'])
        # with open('vocabulary.txt', 'w') as f:
        #     vocab = sorted(vect.vocabulary_.keys())
        #     vocabL = [str(len(w)) for w in vocab if len(w) > 10]
        #     print '\n'.join(vocabL)
        #     vocab = [w for w in vocab if len(w) > 10]
        #     f.write('\n'.join(vocab))

        # tc.grid_search('SVM', data)