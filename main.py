from src.util import load_dataset, trainer
from src.models import logistic_regression,\
    ensemble_classifier,\
    upsampled_logistic_regression,\
    upsampled_dense_model, \
    dense_model, dense
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced

x,y = load_dataset()

model = dense()

trainer = trainer(model,x,y,0.25,'Balanced Generated Dense Model',keras_balanced=False)

trainer.fit_repeat(10)