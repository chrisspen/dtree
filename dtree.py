"""
2012.1.24 CKS
Code for building and using a decision tree for classification or regression.

todo:
+support regression
+support class probabilities for discrete classes
-support matching nearest node element when a new element in the query vector is encountered
-support missing elements in the query vector
-support sparse data sets
"""
from collections import defaultdict
from pprint import pprint
import csv
import math
import os
import cPickle as pickle
import re
import unittest

from decimal import Decimal

VERSION = (0, 1, 0)
__version__ = '.'.join(map(str, VERSION))

def mean(seq):
    return sum(seq)/float(len(seq))

def variance(seq):
    m = mean(seq)
    return sum((v-m)**2 for v in seq)/float(len(seq))

def mean_absolute_error(seq, correct):
    assert len(seq) == len(correct)
    diffs = [abs(a-b) for a,b in zip(seq,correct)]
    return sum(diffs)/float(len(diffs))

class DDist(object):
    """
    Incrementally tracks the probability distribution of discrete elements.
    """
    
    def __init__(self, seq=None):
        self.counts = defaultdict(int)
        self.total = 0
        if seq:
            for k in seq:
                self.counts[k] += 1
                self.total += 1
    
    def __getitem__(self, k):
        """
        Returns the probability for the given element.
        """
        cnt = 0
        if k in self.counts:
            cnt = self.counts[k]
        return cnt/float(self.total)
    
    def add(self, k):
        """
        Increments the count for the given element.
        """
        self.counts[k] += 1
        self.total += 1
                
    def probs(self):
        """
        Returns a list of probabilities for all elements.
        """
        return [
            (k, self.counts[k]/float(self.total))
            for k in self.counts.iterkeys()
        ]
    
    @property
    def best(self):
        """
        Returns the element with the highest probability.
        """
        b = (-1e999999, None)
        for k,c in self.counts.iteritems():
            b = max(b, (c,k))
        return b[1]
    
    def __repr__(self):
        s = []
        for k,prob in self.probs():
            s.append("%s=%s" % (k,prob))
        return "<%s %s>" % (type(self).__name__, ', '.join(s))

class CDist(object):
    """
    Incrementally tracks the probability distribution of continuous numbers.
    """
    
    def __init__(self, seq=None):
        self.mean_sum = 0
        self.mean_count = 0
        self.last_variance = 0
        if seq:
            for n in seq:
                self += n
    
    def __repr__(self):
        return "<%s mean=%s variance=%s>" \
            % (type(self).__name__, self.mean, self.variance)
    
    def __iadd__(self, value):
        last_mean = self.mean
        self.mean_sum += value
        self.mean_count += 1
        if last_mean is not None:
            self.last_variance = self.last_variance \
                + (value  - last_mean)*(value - self.mean)
        return self
        
    @property
    def mean(self):
        if self.mean_count:
            return self.mean_sum/float(self.mean_count)
    
    @property
    def variance(self):
        return self.last_variance/float(self.mean_count)

DEFAULT_ENTROPY_METHOD = 1

def entropy(data, class_attr=None, method=DEFAULT_ENTROPY_METHOD):
    """
    Calculates the entropy of the attribute attr in given data set data.
    
    Parameters:
    data<dict|list> :=
        if dict, treated as value counts of the given attribute name
        if list, treated as a raw list from which the value counts will be generated
    attr<string> := the name of the class attribute
    """
    assert (class_attr is None and isinstance(data,dict)) \
        or (class_attr is not None and isinstance(data,list))
    if isinstance(data, dict):
        counts = data
    else:
        counts = defaultdict(float) # {attr:count}
        for record in data:
            # Note: A missing attribute is treated like an attribute with a value
            # of None, representing the attribute is "irrelevant".
            counts[record.get(class_attr)] += 1.0
    len_data = sum(cnt for _,cnt in counts.iteritems())
    n = max(2, len(counts))
    total = float(sum(counts.values()))
    assert total, "There must be at least one non-zero count."
    try:
        #return -sum((count/total)*math.log(count/total,n) for count in counts)
        if method == 0:
            # Traditional entropy.
            return -sum((count/len_data)*math.log(count/len_data,n)
                for count in counts.itervalues())
        elif method == 1:
            # Modified entropy that down-weights universally unique values.
            return -sum((count/len_data)*math.log(count/len_data,n)
                for count in counts.itervalues()) - ((len(counts)-1)/float(total))
        elif method == 2:
            # Modified entropy that down-weights universally unique values
            # as well as features with large numbers of values.
            return -sum((count/len_data)*math.log(count/len_data,n)
                for count in counts.itervalues()) - 100*((len(counts)-1)/float(total))
        else:
            raise Exception, "Unknown entropy method %s." % method
    except:
        print 'Error:',counts
        raise

DEFAULT_VARIANCE_ENTROPY_METHOD = 1

def entropy_variance(data, class_attr=None,
    method=DEFAULT_VARIANCE_ENTROPY_METHOD):
    """
    Calculates the variance fo a continuous class attribute, to be used as an
    entropy metric.
    """
    assert method in (1,2,)
    assert (class_attr is None and isinstance(data,dict)) \
        or (class_attr is not None and isinstance(data,list))
    if isinstance(data, dict):
        lst = data
    else:
        lst = [record.get(class_attr) for record in data]
    return variance(lst)

def gain(data, attr, class_attr,
    method=DEFAULT_ENTROPY_METHOD,
    only_sub=0, prefer_fewer_values=False, entropy_func=None):
    """
    Calculates the information gain (reduction in entropy) that would
    result by splitting the data on the chosen attribute (attr).
    
    Parameters:
    
    prefer_fewer_values := Weights the gain by the count of the attribute's
        unique values. If multiple attributes have the same gain, but one has
        slightly fewer attributes, this will cause the one with fewer
        attributes to be preferred.
    """
    entropy_func = entropy_func or entropy
    val_freq       = defaultdict(float)
    subset_entropy = 0.0

    # Calculate the frequency of each of the values in the target attribute
    for record in data:
        val_freq[record.get(attr)] += 1.0

    # Calculate the sum of the entropy for each subset of records weighted
    # by their probability of occuring in the training set.
    for val in val_freq.keys():
        val_prob        = val_freq[val] / sum(val_freq.values())
        data_subset     = [record for record in data if record.get(attr) == val]
        e = entropy_func(data_subset, class_attr, method=method)
        subset_entropy += val_prob * e
        
    if only_sub:
        return subset_entropy

    # Subtract the entropy of the chosen attribute from the entropy of the
    # whole data set with respect to the target attribute (and return it)
    main_entropy = entropy_func(data, class_attr, method=method)
    
    # Prefer gains on attributes with fewer values.
    if prefer_fewer_values:
#        n = len(val_freq)
#        w = (n+1)/float(n)/2
        #return (main_entropy - subset_entropy)*w
        return ((main_entropy - subset_entropy), 1./len(val_freq))
    else:
        return (main_entropy - subset_entropy)

def gain_variance(*args, **kwargs):
    """
    Calculates information gain using variance as the comparison metric.
    """
    return gain(entropy_func=entropy_variance, *args, **kwargs)

def majority_value(data, class_attr):
    """
    Creates a list of all values in the target attribute for each record
    in the data list object, and returns the value that appears in this list
    the most frequently.
    """
    if is_continuous(data[0][class_attr]):
        return CDist(seq=[record[class_attr] for record in data])
    else:
        return most_frequent([record[class_attr] for record in data])

def most_frequent(lst):
    """
    Returns the item that appears most frequently in the given list.
    """
    lst = lst[:]
    highest_freq = 0
    most_freq = None

    for val in unique(lst):
        if lst.count(val) > highest_freq:
            most_freq = val
            highest_freq = lst.count(val)
            
    return most_freq

def unique(lst):
    """
    Returns a list made up of the unique values found in lst.  i.e., it
    removes the redundant values in lst.
    """
    lst = lst[:]
    unique_lst = []

    # Cycle through the list and add each value to the unique list only once.
    for item in lst:
        if unique_lst.count(item) <= 0:
            unique_lst.append(item)
            
    # Return the list with all redundant values removed.
    return unique_lst

def get_values(data, attr):
    """
    Creates a list of values in the chosen attribut for each record in data,
    prunes out all of the redundant values, and return the list.  
    """
    return unique([record[attr] for record in data])

def choose_attribute(data, attributes, class_attr, fitness):
    """
    Cycles through all the attributes and returns the attribute with the
    highest information gain (or lowest entropy).
    """
    best = (-1e999999, None)
    for attr in attributes:
        if attr == class_attr:
            continue
        gain = fitness(data, attr, class_attr)
#        print 'attr/gain:',attr,gain
        best = max(best, (gain, attr))
    return best[1]

def is_continuous(v):
    return isinstance(v, (float, Decimal))

def create_decision_tree(data, attributes, class_attr, fitness_func):
    """
    Returns a new decision tree based on the examples given.
    """
    
    data = list(data) if isinstance(data, FileData) else data
    unique_class_values = set(record[class_attr] for record in data)

    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    if not data or (len(attributes) - 1) <= 0:
#        print 'leaf1:',unique_class_values
        #default = majority_value(data, class_attr)
#        return default
        if is_continuous(data[0][class_attr]):
            value = CDist(seq=[r[class_attr] for r in data])
        else:
            value = DDist(seq=[r[class_attr] for r in data])
        return value
    elif len(unique_class_values) == 1:
        # If all the records in the dataset have the same classification,
        # return that classification.
#        print 'leaf2:',value
        value = unique_class_values.pop()
        if is_continuous(data[0][class_attr]):
            value = CDist(seq=[value])
        else:
            value = DDist(seq=[r[class_attr] for r in data])
        return value
    else:
        # Choose the next best attribute to best classify our data
#        print class_attr
#        print attributes
#        print fitness_func
        best = choose_attribute(
            data,
            attributes,
            class_attr,
            fitness_func)

        # Create a new decision tree/node with the best attribute and an empty
        # dictionary object--we'll fill that up next.
        tree = {best:{}}

        # Create a new decision tree/sub-node for each of the values in the
        # best attribute field
        for val in get_values(data, best):
            # Create a subtree for the current value under the "best" field
            subtree = create_decision_tree(
                [r for r in data if r[best] == val],
                [attr for attr in attributes if attr != best],
                class_attr,
                fitness_func)

            # Add the new subtree to the empty dictionary object in our new
            # tree/node we just created.
            tree[best][val] = subtree

    return tree

ATTR_TYPE_NOMINAL = 'nominal'
ATTR_TYPE_DISCRETE = 'discrete'
ATTR_TYPE_CONTINUOUS = 'continuous'

ATTR_MODE_CLASS = 'class'

ATTR_HEADER_PATTERN = re.compile("([^,:]+):(nominal|discrete|continuous)(?::(class))?")

class FileData(object):
    """
    Parses, validates and iterates over tabular data in a file.
    
    This does not store the actual data rows. It only stores the row schema.
    """
    
    def __init__(self, filename=None):
        self.filename = filename
        if filename:
            assert os.path.isfile(filename), \
                "File \"%s\" does not exist." % filename
                
        self.header_types = {} # {attr_name:type}
        self.class_attr_name = None
        self.row_map = [] # [attr_name,...]
                
    def __len__(self):
        if self.filename:
            return max(0, open(self.filename).read().strip().count('\n'))
        return 0

    @property
    def attribute_names(self):
        self._read_header()
        return [
            n for n in self.header_types.iterkeys()
            if n != self.class_attr_name
        ]

    def get_attribute_type(self, name):
        if not self.header_types:
            self._read_header()
        return self.header_types[name]

    @property
    def is_continuous_class(self):
        self._read_header()
        return self.get_attribute_type(self.class_attr_name) \
            == ATTR_TYPE_CONTINUOUS

    def _read_header(self):
        if not self.filename or self.header_types:
            return
        rows = csv.reader(open(self.filename))
        header = rows.next()
        self.header_types = {} # {attr_name:type}
        self.class_attr_name = None
        self.row_map = [] # [attr_name,...]
        for el in header:
            matches = ATTR_HEADER_PATTERN.findall(el)
            assert matches, "Invalid header element: %s" % (el,)
            el_name,el_type,el_mode = matches[0]
            el_name = el_name.strip()
            self.row_map.append(el_name)
            self.header_types[el_name] = el_type
            if el_mode == ATTR_MODE_CLASS:
                assert self.class_attr_name is None, \
                    "Multiple class attributes are not supported."
                self.class_attr_name = el_name
            else:
                assert self.header_types[el_name] != ATTR_TYPE_CONTINUOUS, \
                    "Non-class continuous attributes are not supported."
        assert self.class_attr_name, "A class attribute must be specified."

    def __iter__(self):
        if not self.filename:
            return
        self._read_header()
        try:
            rows = csv.reader(open(self.filename))
            header = rows.next()
            while 1:
                _row = rows.next()
                if not _row:
                    continue
                row = []
                for el_name, el_value in zip(self.row_map, _row):
                    if self.header_types[el_name] == ATTR_TYPE_DISCRETE:
                        row.append(int(el_value))
                    elif self.header_types[el_name] == ATTR_TYPE_CONTINUOUS:
                        row.append(float(el_value))
                    else:
                        row.append(el_value)
                yield dict(zip(self.row_map,row))
        except StopIteration:
            pass

USE_NEAREST = 'use_nearest'
MISSING_VALUE_POLICIES = set([
    USE_NEAREST,
])

class DTree(object):
    """
    Wrapper for a decision tree allowing either classification or regression.
    """
    
    def __init__(self):
        self._tree = None
        self._data = None
        self.missing_value_policy = {}
        
    def set_missing_value_policy(self, policy, target_attr_name=None):
        """
        Sets the behavior for one or all attributes to use when traversing the
        tree using a query vector and it encounters a branch that does not
        exist.
        """
        assert policy in MISSING_VALUE_POLICIES, \
            "Unknown policy: %s" % (policy,)
        for attr_name in self._data.attribute_names:
            if target_attr_name is not None and target_attr_name != attr_name:
                continue
            self.missing_value_policy[attr_name] = policy
    
    def save(self, fn):
        pickle.dump(self, open(fn,'w'))
    
    @classmethod
    def load(cls, fn):
        tree = pickle.load(open(fn))
        assert isinstance(tree, cls), "Invalid pickle."
        return tree
    
    def _classify(self, data):
        """
        Returns a list of classifications for each of the records in the data
        list as determined by the given decision tree.
        """
        for record in data:
            assert isinstance(record, dict)
            yield self._get_leaf(record)
    
    def _get_leaf(self, record):
        """
        This function recursively traverses the decision tree and returns a
        classification for the given record.
        """
        misses = 0
        tree = self._tree
        while 1:
            # If the current node is a string, then we've reached a leaf node 
            # and we can return it as our answer.
            if not isinstance(tree, dict):
                return tree
            else:
                # Traverse the tree further until a leaf node is found.
                assert isinstance(tree, dict), \
                    "Invalid type: %s" % (type(tree),)
                attr = tree.keys()[0]
                if record[attr] in tree[attr]:
                    tree = tree[attr][record[attr]]
                else:
                    # The tree does not contain a branch for the value in the
                    # query vector, so attempt a work-around according to our
                    # policy.
                    misses += 1
                    policy = self.missing_value_policy.get(attr)
                    assert policy, \
                        ("Node %s has no key value %s. " +
                         "Available keys are %s.") % (
                        attr, record[attr],
                        ', '.join(map(str, tree[attr].keys())))
                    if policy == USE_NEAREST:
                        assert self._data.header_types[attr] \
                            in (ATTR_TYPE_DISCRETE, ATTR_TYPE_CONTINUOUS), \
                            "The use-nearest policy is invalid for nominals."
                        nearest = (1e999999, None)
#                        print 'actual:',record[attr]
#                        print 'keys:',tree[attr].keys()
                        for value in tree[attr].keys():
                            nearest = min(
                                nearest,
                                (abs(value - record[attr]), value))
                        _,nearest_value = nearest
                        tree = tree[attr][nearest_value]
                    else:
                        raise Exception, "Unknown policy: %s" % (policy,)
    
    def _regress(self, data):
        """
        Returns a list of regressions for each of the records in the data
        list as determined by the given decision tree.
        """
        for record in data:
            assert isinstance(record, dict)
            yield self._get_leaf(record)
    
    def predict(self, data):
        """
        Returns a classification or regression for the given data set
        depending on the class attribute type.
        
        Accepts a list of records, either as an explicit list or FileData
        object, or a single record in the form of a dictionary.
        """
        if isinstance(data, (list, FileData)):
            if self._data.is_continuous_class:
                itr = self._regress(data)
            else:
                itr = self._classify(data)
            return list(itr)
        else:
            assert isinstance(data, dict)
            if self._data.is_continuous_class:
                return self._regress([data]).next()
            else:
                return self._classify([data]).next()
    
    def test(self, data):
        """
        Iterates over the data, classifying or regressing each element and then
        finally returns the classification accuracy or mean-absolute-error.
        """
#        assert data.header_types == self._data.header_types, \
#            "Test data schema does not match the tree's schema."
        is_continuous = self._data.is_continuous_class
        agg = CDist()
        for record in data:
            actual_value = self.predict(record)
            expected_value = record[self._data.class_attr_name]
            if is_continuous:
                actual_value = actual_value.mean
                agg += abs(actual_value - expected_value)
            else:
                if isinstance(actual_value, DDist):
#                    print actual_value[actual_value.best]
                    actual_value = actual_value.best
                agg += actual_value == expected_value
        return agg
    
    @classmethod
    def build(cls, data, *args, **kwargs):
        """
        Constructs a classification or regression tree in a single batch by
        analyzing the given data.
        """
        assert isinstance(data, FileData)
        if data.is_continuous_class:
            fitness_func = gain_variance
        else:
            fitness_func = gain
        
        t = cls(*args, **kwargs)
        t._tree = create_decision_tree(
            data=data,
            attributes=data.attribute_names,
            class_attr=data.class_attr_name,
            fitness_func=fitness_func,
        )
        t._data = data
        return t
    
class Test(unittest.TestCase):

    def test_stat(self):
        nums = range(1,10)
        s = CDist()
        seen = []
        for n in nums:
            seen.append(n)
            s += n
            print 'mean:',s.mean
            print 'variance:',variance(seen)
            print 'variance:',s.variance
        self.assertAlmostEqual(s.mean, mean(nums), 1)
        self.assertAlmostEqual(s.variance, variance(nums), 2)

    def test_data(self):
        data = FileData('rdata1')
        self.assertEqual(len(data), 16)
        data = list(FileData('rdata1'))
        self.assertEqual(len(data), 16)
        for row in data:
            print row
    
    def test_tree(self):
        t = DTree.build(FileData('rdata2'))
        pprint(t._tree, indent=4)
        result = t.test(FileData('rdata1'))
        print 'MAE:',result.mean
        self.assertAlmostEqual(result.mean, 0.001368, 5)
        
        t = DTree.build(FileData('cdata1'))
        pprint(t._tree, indent=4)
        result = t.test(FileData('cdata1'))
        print 'Accuracy:',result.mean
        self.assertAlmostEqual(result.mean, 1.0, 5)
        
        t = DTree.build(FileData('cdata2'))
        pprint(t._tree, indent=4)
        result = t.test(FileData('cdata3'))
        print 'Accuracy:',result.mean
        self.assertAlmostEqual(result.mean, 0.75, 5)
        
        t = DTree.build(FileData('cdata4'))
        pprint(t._tree, indent=4)
        result = t.test(FileData('cdata4'))
        print 'Accuracy:',result.mean
        self.assertAlmostEqual(result.mean, 0.5, 5)
        
        # Send it a case it's never seen.
        try:
            # By default, it should throw an exception.
            t.predict(dict(a=1,b=2,c=3,d=4))
            self.assertTrue(0)
        except AssertionError:
            pass
        # But if we tell it to use the nearest value, then it should pass.
        t.set_missing_value_policy(USE_NEAREST)
        result = t.predict(dict(a=1,b=2,c=3,d=4))
        print result

if __name__ == '__main__':
    unittest.main()
    