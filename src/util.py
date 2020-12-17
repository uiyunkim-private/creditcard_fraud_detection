import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,Normalizer,LabelBinarizer
from src.metric import evaluator
from tqdm import tqdm
import time
from imblearn.keras import balanced_batch_generator
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE,ADASYN

import os
def load_dataset():
    csv = pd.read_csv('dataset/creditcard.csv')
    y = np.array(csv['Class'])
    x = csv.drop(['Class','Time'],axis=1)
    return x,y

def standardize(x):
    return StandardScaler().fit_transform(x)

def normalize(x):
    return Normalizer().fit_transform(x)

class trainer:
    def __init__(self,model,x,y,test_size,name,keras_balanced=False):
        self.model = model
        self.name = name
        self.x = x
        self.y = y
        self.test_size = test_size
        self.keras_balanced = keras_balanced
        self.__preprocess()

    def __preprocess(self):
        self.x = standardize(self.x)
        self.x = normalize(self.x)

        self.y = LabelBinarizer().fit_transform(self.y)
        if self.keras_balanced:
            self.keras_generator, self.steps_per_epoch = balanced_batch_generator(self.x, self.y, sampler=NearMiss(version=3), batch_size=32)

    def __train_test_split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=self.test_size,shuffle=True)

    def fit_result(self):
        self.__train_test_split()

        if self.keras_balanced:
            self.model.fit(x=self.keras_generator,steps_per_epoch=10000, epochs=10 )

        else:
            self.model.fit(self.x_train,self.y_train)

        history = evaluator(self.model,self.x_test,self.y_test,self.name)
        #history.draw_precision_recall_curve()
        history.plot_confusion_matrix()

        return history.get_accuracy_score(), \
               history.get_balanced_score(), \
               history.get_f1_score(), \
               history.get_precision_score(), \
               history.get_recall_score()

    def fit_repeat(self,num_iter):
        recall_score_records = []
        f1_score_records = []
        balanced_score_records = []

        print("Repeating fit and Collecting results.")
        time.sleep(1)
        for _ in tqdm(range(num_iter)):
            accuracy_score, balanced_score, f1_score, precision_score, recall_score = self.fit_result()
            recall_score_records.append(recall_score)
            f1_score_records.append(f1_score)
            balanced_score_records.append(balanced_score)

        print("Consistency result with test_zie",self.test_size,recall_score_records)
        plt.hlines(sum(recall_score_records) / num_iter,xmin=0,xmax=num_iter-1,colors='r',linestyles='dashed')
        plt.plot(recall_score_records,'r')

        plt.hlines(sum(f1_score_records) / num_iter, xmin=0, xmax=num_iter-1,colors='g',linestyles='dashed')
        plt.plot(f1_score_records, 'g')

        plt.hlines(sum(balanced_score_records) / num_iter, xmin=0, xmax=num_iter-1,colors='k',linestyles='dashed')
        plt.plot(balanced_score_records, 'k')

        plt.legend(['recall', 'f1','balanced'], loc='lower right')
        plt.ylim(0,1)
        plt.title('train_test_split = '+str(self.test_size) + ' repeated ' + str(num_iter) + '\n'+
                  'Ave: recall('+
                  str(int(sum(recall_score_records)/len(recall_score_records) *100) )+') f1('+
                  str(int(sum(f1_score_records)/len(f1_score_records) *100) )  + ') balanced('+
                  str(int(sum(balanced_score_records)/len(balanced_score_records)*100) )+')' + '\n' +
                  'Max: recall(' +
                  str(int(max(recall_score_records)*100)) + ') f1(' +
                  str(int(max(f1_score_records)*100)) + ') balanced(' +
                  str(int(max(balanced_score_records)*100)) + ')'
                  )
        if not os.path.exists('figure/'+self.name):
            os.makedirs('figure/'+self.name)

        plt.savefig('figure/'+self.name+'/'+'train_test_split = '+str(self.test_size) + ' repeated ' + str(num_iter) + '.png')
        plt.show()
