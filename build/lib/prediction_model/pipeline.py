from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from config import config
import prediction_model.preprocessing.preprocessing as pp
from sklearn.preprocessing import MinMaxScaler
import numpy as np

classification_pipeline = Pipeline(
    [
        ('MeanImputation', pp.MeanImputer(variables=config.NUM_FEATURES)),
        ('ModeImputation', pp.ModeImputer(variables=config.CAT_FEATURES)),
        ('DomainProcessing', pp.DomainProcessing(variable_to_modify=config.FEATURE_TO_MODIFY,variable_to_add=config.FEATURE_TO_ADD)),
        ('DropFeatures', pp.DropColumns(variables_to_drop=config.DROP_FEATURES)),
        ('LabelEncoding', pp.CustomLabelEncoder(variables=config.FEATURES_TO_ENCODE)),
        ('LogTransformation', pp.LogTransformation(variables=config.LOG_FEATURES)),
        ('MinMaxScale', MinMaxScaler()),
        ('LogisticClassifier', LogisticRegression(random_state=0))
    ]
)
