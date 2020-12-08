import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_precision_recall_curve,recall_score, f1_score, precision_score,average_precision_score,plot_confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import multiprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifier
# VotingClassifier
from sklearn.ensemble import VotingClassifier
def load_dataset():
    csv = pd.read_csv('creditcard.csv')
    y = np.array(csv['Class'])
    x = csv.drop(['Class','Time'],axis=1)
    return x,y

def get_scores(x,y,classifier,test_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
    classifier.fit(x_train, y_train)
    return recall_score(y_test, classifier.predict(x_test)),\
           f1_score(y_test, classifier.predict(x_test)),\
           precision_score(y_test, classifier.predict(x_test)), \
           balanced_accuracy_score(y_test,classifier.predict(x_test))

# def analyze(x,y,classifier,test_size):
#     results = {}
#     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
#     classifier.fit(x_train, y_train)
#
#     results.update({'accuracy_score':accuracy_score(y_test,classifier.predict(x_test)),
#                     "precision_score":precision_score(y_test,classifier.predict(x_test)),
#                     "f1_score":f1_score(y_test,classifier.predict(x_test)),
#                     "recall_score":recall_score(y_test, classifier.predict(x_test))
#                     })
#
#     plot_confusion_matrix(classifier,x_test,y_test)
#     plt.savefig('cm.PNG')
#     confusion_matrix_image = Image.open('cm.PNG')
#
#     # y_score = classifier.decision_function(x_test)
#     # average_precision = average_precision_score(y_test, y_score)
#     # disp = plot_precision_recall_curve(classifier, x_test, y_test)
#     # disp.ax_.set_title('2-class Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))
#     # plt.savefig('prc.PNG')
#     precision_recall_curve_image = Image.open('prc.PNG')
#
#     results.update({'confusion_matrix_image':confusion_matrix_image,
#                     'precision_recall_curve_image':precision_recall_curve_image})
#     return results

def select_test_split_size(x,y,classifier,num_iter):
    for test_size in [0.25]:
        recall_score_records = []
        f1_score_records = []
        precision_score_records = []
        balanced_score_records = []
        for _ in tqdm(range(num_iter)):
            recall, f1, precision, balanced = get_scores(x, y, classifier, test_size=test_size)
            recall_score_records.append(recall)
            f1_score_records.append(f1)
            precision_score_records.append(precision)
            balanced_score_records.append(balanced)
        print("Consistency result with test_zie",test_size,recall_score_records)
        plt.hlines(sum(recall_score_records) / num_iter,xmin=0,xmax=num_iter-1,colors='r',linestyles='dashed')
        plt.plot(recall_score_records,'r')

        plt.hlines(sum(f1_score_records) / num_iter, xmin=0, xmax=num_iter-1,colors='g',linestyles='dashed')
        plt.plot(f1_score_records, 'g')

        # plt.hlines(sum(precision_score_records) / num_iter, xmin=0, xmax=num_iter-1,colors='b',linestyles='dashed')
        # plt.plot(precision_score_records, 'b')

        plt.hlines(sum(balanced_score_records) / num_iter, xmin=0, xmax=num_iter-1,colors='k',linestyles='dashed')
        plt.plot(balanced_score_records, 'k')

        plt.legend(['recall', 'f1','balanced'], loc='lower right')
        plt.ylim(0,1)
        plt.title('train_test_split = '+str(test_size) + ' repeated ' + str(num_iter))
        plt.savefig('train_test_split = '+str(test_size) + ' repeated ' + str(num_iter) + '.png')
        plt.show()
models = [
    ('ada', AdaBoostClassifier()),
    ('bc', BaggingClassifier(n_jobs=-1)),
    ('etc',ExtraTreesClassifier(n_jobs=-11)),
    ('gbc', GradientBoostingClassifier()),
    ('rfc', RandomForestClassifier(n_jobs=-1)),
    ('knn', KNeighborsClassifier(n_jobs=-1)),
    ('svc', SVC(probability=True)),
    ('dtc', DecisionTreeClassifier()),
    ('lr', LogisticRegressionCV(n_jobs=-1)),
    #('ridge', RidgeClassifier()),
]

x,y = load_dataset()
x = StandardScaler().fit_transform(x)
x = Normalizer().fit_transform(x)

classifier = VotingClassifier(models, voting='soft',n_jobs=-1)
#classifier = make_pipeline(PolynomialFeatures(),LogisticRegression())
# classifier = LogisticRegressionCV(n_jobs=-1,max_iter=9999)
select_test_split_size(x,y,classifier,10)
#result = analyze(x,y,classifier,0.2)





