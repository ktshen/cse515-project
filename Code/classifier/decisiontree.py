from .classifier import Classifier
import numpy as np


class DecisionTree(Classifier):
    def __init__(self, dimension=None, visualization=False):
        self._visualization = visualization
        
        self.max_depth = 3
        self.min_size = 10
        self.tree = None
        
        pass

    def fit(self, data, gt):
        # The data format should be:
        # data = [[features of data1], [features of data2], ... [features of dataN]]
        # gt = [y1, y2, y3, y4 ...] for each yi is true / false.
        data = np.c_[data,gt]
        self.tree  = self.build_tree(data)
#        self.print_tree(self.tree)
        pass

    def predict(self, data):
        if self.tree is None:
            raise Exception("Please train Decision Tree model first")
        predictval = []
        for d in data:
            predictval.append(bool(self.predicteach(self.tree,d)))
        return predictval
        pass
    
    # build a decision tree
    def build_tree(self, data):
        root = self.get_split(data)
        self.split(root, 1)
        return root
    
#    split a data based on an attribute and an value
    def test_split(self, index, value, data):
        left, right = list(), list()
        for row in data:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right
    
#    calculate the gini index for the splitting data
    def gini_index(self, groups, classes):
        # calculate all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        gini = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
        return gini
    # get the best split point for the data
    def get_split(self,data):
        class_values = list(set(row[-1] for row in data))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(data[0])-1):
            for row in data:
                groups = self.test_split(index, row[index], data)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
        return {'index':b_index, 'value':b_value, 'groups':b_groups}
    
    # create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)
    
    # Create child splits for a node or make terminal
    def split(self, node, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth+1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth+1)
    
    def predicteach(self, tree, data):
        if data[tree['index']] < tree['value']:
            if isinstance(tree['left'], dict):
                return self.predicteach(tree['left'], data)
            else:
                return tree['left']
        else:
            if isinstance(tree['right'], dict):
                return self.predicteach(tree['right'], data)
            else:
                return tree['right']
    
#    def print_tree(node, depth=0):
#       if isinstance(node, dict):
#           print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
#           print_tree(node['left'], depth+1)
#           print_tree(node['right'], depth+1)
#       else:
#           print('%s[%s]' % ((depth, node)))
    
