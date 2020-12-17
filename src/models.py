from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import VotingClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

def dense():
    model = Sequential()
    model.add(Dense(256,input_dim=29))
    model.add(Dense(512))
    model.add(Dense(1024))
    model.add(Dense(2048))
    model.add(Dense(2048))
    model.add(Dense(1024))
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model

def dense_model():
    model = convert_keras_classifier(dense, epochs=10, batch_size=2048)
    return model


def convert_keras_classifier(model,epochs,batch_size):
    model = KerasClassifier(build_fn=model,epochs=epochs,batch_size=batch_size)
    return model

def logistic_regression():
    return LogisticRegressionCV(max_iter=9999)

def upsampled_dense_model():
    model = convert_keras_classifier(dense, epochs=20, batch_size=2048)
    model = make_pipeline(SMOTE(sampling_strategy='minority', n_jobs=-1),
                           model)
    return model

def upsampled_logistic_regression():
    model = logistic_regression()
    model = make_pipeline(SMOTE(sampling_strategy='minority', n_jobs=-1),
                           # NearMiss(sampling_strategy='majority',n_jobs=-1),
                           model)
    return model

def ensemble_classifier():
    models = [
        ('rfc', RandomForestClassifier(n_jobs=-1)),
        ('knn', KNeighborsClassifier(n_jobs=-1)),
        ('svc', SVC(probability=True, max_iter=9999)),
        ('dtc', DecisionTreeClassifier()),
        ('lr', LogisticRegressionCV(n_jobs=-1, max_iter=9999)),
    ]
    model = VotingClassifier(models, voting='soft',n_jobs=-1)
    return model