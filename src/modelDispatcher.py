from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xml.etree.ElementTree import *
from gradboostWrapper import GradBoostWrapper

import numpy as np

class model_dispatcher:

    def __init__(self, model, feature_list):

        try:
            assert type(model) == AdaBoostClassifier

            self.model_type = 'BDT'
            self.model = model
            self.xml_tree = None

        except AssertionError:
            raise TypeError('Model type not supported.')

        try:
            assert self.model.estimators_[0].n_features_ == len(feature_list)
            self.feature_list = feature_list

        except AssertionError:
            raise TypeError('Feature list has different number of features to model.')



    # Sends model to BDT wrapper to crete XML tree.
    def dispatch(self):
        if self.model_type == 'BDT':
            wrapper = GradBoostWrapper(self.model, self.feature_list)
            self.xml_tree = wrapper.build()
        else:
            pass


    def to_file(self, path):
        self.xml_tree.write(open(path, 'w'), xml_declaration=True)