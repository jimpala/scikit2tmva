from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xml.etree.ElementTree import *
import sklearn.tree


class GradBoostWrapper:

    def __init__(self, bdt, element_list):
        self.bdt = bdt
        self.element_list = element_list
        # Create dummy tree for value queries.
        self.dummy_tree = self.bdt.estimators_[0]

        self.root = None

    def build(self):

        self.root = Element('MethodSetup', attrib={'Method': 'BDT::BDT_scikit2tmva'})

        # Build from root's direct children
        ############
        dump(self.root)
        # <General>
        self.root.append(ElementTree(file="../boilerplate/GeneralInfo.xml").getroot())
        dump(self.root)
        # <Options>
        self.root.append(ElementTree(file="../boilerplate/Options.xml").getroot())
        dump(self.root)
        # <Variables>
        self.root.append(self.generate_variables())
        dump(self.root)
        # <Spectators>
        self.root.append(Element('Spectators', attrib={'NSpec': '0'}))
        dump(self.root)
        # <Classes>
        self.root.append(self.generate_classes())
        dump(self.root)
        # <Transforms>
        self.root.append(Element('Transformations', attrib={'NTransformations': '0'}))
        dump(self.root)
        # <MVAPdfs>
        self.root.append(Element('MVAPdfs'))
        dump(self.root)
        # <Weights>
        self.root.append(self.generate_weights())
        dump(self.root)

        return ElementTree(element=self.root)

    def generate_variables(self):
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
    def generate_classes(self):
        return ElementTree(file="../boilerplate/Classes.xml").getroot()


    # Use tuple stack with (my_index, parent_index), along with list of elements.
    def generate_weights(self):

        # Get the estimator list from AdaBoostClassifier.
        estimator_list = self.bdt.estimators_
        n_estimators = len(estimator_list)

        # The root tag here (direct child of overall document root node) is <Weights>.
        weights = Element('Weights', attrib={'NTrees': '{:d}'.format(n_estimators),
                                             'AnalysisType': '0'})

        # Iterate through all the estimators.
        for estimator, est_i in zip(estimator_list, range(n_estimators)):

            # Create <BinaryTree> tag for current estimator, with weights attribute.
            this_weight = self.bdt.estimator_weights_[est_i]
            this_tree = SubElement(weights, 'BinaryTree', attrib={'type': 'DecisionTree',
                                                                  'boostWeight': '{:.8e}'.format(this_weight),
                                                                  'itree': '{:d}'.format(est_i)})

            # with open("test.dot", 'w') as f:
            #     f = sklearn.tree.export_graphviz(estimator, out_file=f)

            # Stack is a queuing system, filled dict 'structs'
            # with index, parent index, position and depth info.
            # Tags is a dict to house elements by index number.
            stack = []
            tags = dict()

            # Get the underlying tree object for the current estimator.
            t = estimator.tree_

            # Create the root <Node> tag
            tags[0] = (SubElement(this_tree, 'Node', attrib={'pos': 's',
                                                             'depth': '0',
                                                             'NCoef': '0',
                                                             'IVar' : '{:d}'.format(t.feature[0]),
                                                             'Cut': '{:.8e}'.format(t.threshold[0]),
                                                             'cType': '0',
                                                             'res': '{:.8e}'.format(9.9),
                                                             'rms': '{:.4e}'.format(0.0),
                                                             'purity': '{:8e}'.format(t.impurity[0]),
                                                             'nType': '0'}))
            # Root <Node>'s children gets the stack rolling.
            stack.append({'index': t.children_left[0],
                          'parent_i': 0,
                          'pos': 'l',
                          'depth': 1})
            stack.append({'index': t.children_right[0],
                          'parent_i': 0,
                          'pos': 'r',
                          'depth': 1})

            # Iterate through each new tree node in stack.
            while stack:

                # Get the main feature info for this node.
                feature = t.feature[stack[0]['index']]
                cut = t.threshold[stack[0]['index']]
                impurity = t.impurity[stack[0]['index']]

                l_child_i = t.children_left[stack[0]['index']]
                r_child_i = t.children_right[stack[0]['index']]

                # Children L/R value of -1 corresponds to no L or R children for this node.
                # If both are -1, this is a leaf node, and this clause is entered.
                if l_child_i == -1 and r_child_i == -1:
                    feature = -1  # IVar value is -1 for leaf in TMVA.

                    # Value parameter is an 1x2 ndarray with [[prob_back, prob_sig]].
                    # Using enumerate, find out which probability is bigger, then
                    # assign that as the classification of this leaf.
                    sb_probs = t.value[stack[0]['index']][0]
                    s_or_b = [a for a, b in enumerate(sb_probs) if b == max(sb_probs)][0]
                    n_type = 1 if s_or_b == 1 else -1

                # If not a leaf, classification is zero (n/a).
                else:
                    n_type = 0

                # Add the current tag to the tags dict for this tree.
                tags[stack[0]['index']] = (SubElement(tags[stack[0]['parent_i']], 'Node',
                                                      attrib={'pos': '{}'.format(stack[0]['pos']),
                                                              'depth': '{:d}'.format(stack[0]['depth']),
                                                              'NCoef': '0',
                                                              'IVar': '{:d}'.format(feature),
                                                              'Cut': '{:.8e}'.format(cut),
                                                              'cType': '0',
                                                              'res': '{:.8e}'.format(9.9),
                                                              'rms': '{:.4e}'.format(0.0),
                                                              'purity': '{:.8e}'.format(impurity),
                                                              'nType': '{:d}'.format(n_type)}))

                # Append to stack any children.
                if l_child_i != -1:
                    stack.append({'index': l_child_i,
                                  'parent_i': stack[0]['index'],
                                  'pos': 'l',
                                  'depth': stack[0]['depth'] + 1})
                if r_child_i != -1:
                    stack.append({'index': r_child_i,
                                  'parent_i': stack[0]['index'],
                                  'pos': 'r',
                                  'depth': stack[0]['depth'] + 1})

                # Pop front element of stack.
                stack.pop(0)

        return weights


class TreeNode:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left  = left
        self.right = right

    def __str__(self):
        return str(self.data)

