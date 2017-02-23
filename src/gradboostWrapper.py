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
        spectators = Element('Spectators', attrib={'NSpec': '0'})
        classes = self.generate_classes(bdt)
        transforms = Element('Transformations', attrib={'NTransformations': '0'})
        mva_pdfs = Element('MVAPdfs')
        weights = self.generate_weights()


    # {:.13e}
    def generate_variables(self, bdt):
        n_features = self.dummy_tree.n_features_

        variables = Element('Variables', attrib={'NVar': '{:d}'.format(n_features)})

        for index, label in zip(range(n_features), self.element_list):
            SubElement(variables, 'Variable', attrib={'VarIndex': "{:d}".format(index),
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


    # Use tuple stack with (my_index, parent_index), along with list of elements.
    def generate_weights(self, bdt):

        estimator_list = bdt.estimators_
        n_estimators = len(estimator_list)

        weights = Element('Weights', attrib={'NTrees': '{:d}'.format(n_estimators),
                                             'AnalysisType': '0'})
        for estimator, i in zip(estimator_list, range(n_estimators)):

            weight = bdt.estimator_weights_[i]
            this_tree = SubElement(weights, 'BinaryTree', attrib={'type': 'DecisionTree',
                                                                  'boostWeight': '{:.13e}'.format(weight),
                                                                  'itree': '{:d}'.format(i)})

            stack = []
            tags = []

            t = estimator.tree_

            # Root tag.
            tags.append(SubElement(this_tree, 'Node', attrib={'pos': 's',
                                                              'depth': '0',
                                                              'NCoef': '0',
                                                              'IVar' : '{:d}'.format(t.feature[0]),
                                                              'Cut': '{:.17e}'.format(t.threshold[0]),
                                                              'cType': '0',
                                                              'res': '{:.17e}'.format(9.9),
                                                              'rms': '{:17e}'.format(0.0),
                                                              'purity': '{:17e}'.format(t.impurity[0]),
                                                              'nType': '0'}))

            
            stack.append({'index': t.children_left[0],
                          'parent_index': 0,
                          'pos': 'l',
                          'depth': 1})

            stack.append({'index': t.children_right[0],
                          'parent_i': 0,
                          'pos': 'r',
                          'depth': 1})

            for node in stack:

                feature = t.feature[node['index']]
                cut = t.threshold[node['index']]
                impurity = t.impurity[node['index']]

                l_child_i = t.children_left
                r_child_i = t.children_right

                if l_child_i == -1 and r_child_i == -1:
                    feature = -1
                    n_type = 0

                else:
                    sb_probs = t.value[node['index']][0]
                    s_or_b = [a for a, b in enumerate(sb_probs) if b == max(sb_probs)]
                    n_type = 1 if s_or_b == 1 else -1

                tags.append(SubElement(tags[node['parent_i']], 'Node', attrib={'pos': '{}'.format(node['pos']),
                                                                               'depth': '0',
                                                                               'NCoef': '0',
                                                                               'IVar': '{:d}'.format(feature),
                                                                               'Cut': '{:.17e}'.format(cut),
                                                                               'cType': '0',
                                                                               'res': '{:.17e}'.format(9.9),
                                                                               'rms': '{:17e}'.format(0.0),
                                                                               'purity': '{:17e}'.format(impurity),
                                                                               'nType': '{:d}'.format(n_type)}))
                if l_child_i != -1:
                    stack.append({'index': l_child_i,
                                  'parent_index': node['index'],
                                  'pos': 'l',
                                  'depth': node['depth'] + 1})
                if r_child_i != -1:
                    stack.append({'index': r_child_i,
                                  'parent_index': node['index'],
                                  'pos': 'r',
                                  'depth': node['depth'] + 1})
        return weights


class TreeNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left  = left
        self.right = right

    def __str__(self):
        return str(self.data)

