from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xml.etree.ElementTree import *

import numpy as np


class GradBoostWrapper:


    def __init__(self, bdt, element_list):
        self.bdt = bdt
        self.element_list = element_list



    def bdt_builder(self, element_tree, bdt):

        # Create dummy tree for value queries.
        self.dummy_tree = element_tree.estimators_[0]

        self.method = Element('MethodSetup', attrib={'Method': 'BDT::BDT_scikit2tmva'})

        general = ElementTree(file="../boilerplate/GeneralInfo.xml").getroot()
        options = ElementTree(file="../boilerplate/Options.xml")
        variables = self.generate_variables(bdt)
        spectators = Element('Spectators', {'NSpec': '0'})
        classes = self.generate_classes(bdt)
        transforms = Element('Transformations', {'NTransformations': '0'})
        mva_pdfs = Element('MVAPdfs')
        weights = None


    # {:.13e}
    def generate_variables(self, bdt):
        n_features = self.dummy_tree.n_features_

        variables = Element('Variables', {'NVar': '{}'.format(str(n_features))})

        for index, label in zip(range(n_features), self.element_list):
            SubElement(variables, 'Variable',
                       {'VarIndex': "{}".format(str(index)),
                        'Expression': label,
                        'Label': label,
                        'Title': label,
                        'Internal': label,
                        'Type': 'F',
                        'Min': '',
                        'Max': ''})

        return variables

    # For now use boilerplate.
    def generate_classes(self, bdt):
        return ElementTree(file="../boilerplate/Classes.xml")

    def generate_weights(self, bdt):
        pass


class TreeNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left  = left
        self.right = right

    def __str__(self):
        return str(self.data)

