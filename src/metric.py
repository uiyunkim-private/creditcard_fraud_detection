from sklearn.metrics import accuracy_score,confusion_matrix, plot_precision_recall_curve,recall_score, f1_score, precision_score,plot_confusion_matrix, balanced_accuracy_score
import matplotlib.pyplot as plt
import os
import seaborn as sns
class evaluator:
    def __init__(self,model,x_test,y_test,name):
        self.name = name
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def get_accuracy_score(self):
        return accuracy_score(self.y_test, self.model.predict(self.x_test).round())

    def get_precision_score(self):
        return precision_score(self.y_test, self.model.predict(self.x_test).round())

    def get_recall_score(self):
        return recall_score(self.y_test, self.model.predict(self.x_test).round())

    def get_f1_score(self):
        return f1_score(self.y_test, self.model.predict(self.x_test).round())

    def get_balanced_score(self):
        return balanced_accuracy_score(self.y_test, self.model.predict(self.x_test).round())

    def plot_confusion_matrix(self):
        sns.heatmap(confusion_matrix(self.y_test,self.model.predict(self.x_test).round()), annot=True)
        plt.title(" confusion matrix")
        if not os.path.exists('figure/'+self.name):
            os.makedirs('figure/'+self.name)
        plt.savefig('figure/'+self.name+'/'+"confusion matrix.png")
        plt.show()

    def draw_precision_recall_curve(self):
        plot_precision_recall_curve(self.model, self.x_test, self.y_test)
        plt.title(" precision recall curve")
        if not os.path.exists('figure/'+self.name):
            os.makedirs('figure/'+self.name)
        plt.savefig('figure/'+self.name+'/'+"precision recall curve.png")
        plt.show()