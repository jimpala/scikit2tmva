from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xml.etree import ElementTree

import numpy as np

class weights_dispatcher:

    def __init__(self, model):

            try:
                assert type(model) == AdaBoostClassifier

                self.model_type = 'BDT'
                self.model = model

            except AssertionError:
                raise TypeError('Model type not supported.')