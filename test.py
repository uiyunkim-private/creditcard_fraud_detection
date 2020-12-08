import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, plot_precision_recall_curve,recall_score, f1_score, precision_score,average_precision_score,plot_confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm



csv = pd.read_csv('creditcard.csv')
print(csv.groupby(['Class']).size())
print(csv.groupby(['Class']).size().plot(kind = 'bar'))
plt.ylim(0,1000)
plt.show()