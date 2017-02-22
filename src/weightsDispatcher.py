from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xml.etree.ElementTree import *

import numpy as np

class weights_dispatcher:

    def __init__(self, model):

        try:
            assert type(model) == AdaBoostClassifier

            self.model_type = 'BDT'
            self.model = model
            self.xml_tree = None

        except AssertionError:
            raise TypeError('Model type not supported.')


    # Sends model to
    def dispatch(self):
        pass

    def to_file(self, path):
        pass