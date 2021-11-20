import pandas as pa
from math import log
from collections import OrderedDict

def ID3(data, label):
    rootNode = getRootNode(data, label)
    tree = (rootNode, getAttribute(data, label, rootNode))
    return tree

def getAttribute(data, label, rootNode):
    tree = {}
    unique_values = data[rootNode].unique()
    defaultData = data[label].value_counts().idxmax()
    tree.update({'default': defaultData})
    for unique_value in unique_values:
        df = data[data[rootNode] == unique_value]
        info = information(df[label])
        if info > 0:
            df = df.drop([rootNode], axis=1)
            aNode = getRootNode(df, label)
            tup = (aNode, getAttribute(df, label, aNode))
            tree.update({str(unique_value): tup})
        else:
            d = df.filter(items=[label, rootNode]).values[0]
            key = str(d[1])
            value = d[0]
            tree.update({key: value})
    print("dict: ", dict(OrderedDict(sorted(tree.items(), key=lambda d: d[0]))))
    return dict(OrderedDict(sorted(tree.items(), key=lambda d: d[0])))

def getRootNode(data, label):
    gainDict = dict()
    info = information(data[label])
    if info > 0:
        for key in data.keys():
            if key != label:
                df = data.filter(items=[label, key])
                gainDict.update({key: gain(df, key, label, info)})
    root = max(gainDict.keys(), key=(lambda x: gainDict[x]))
    print("root node: ", root )
    return root

def gain(data, key, label, I_total):
    data = pa.DataFrame(data=data)
    entropy_total = 0.0
    unique_values = data[key].unique()
    for value in unique_values:
        df = data[data[key] == value]
        info_value = information(df[label])
        entropy_value = entropy(s=len(df), s_total=len(data), info=info_value)
        entropy_total += entropy_value
    return float(format((I_total - entropy_total), '.5f'))

def entropy(s, s_total, info):
    s = abs(s)
    s_total = abs(s_total)
    if s != 0 and s_total != 0:
        result = (s / s_total) * info
    else:
        result = 0
    return float(format(result, '.5f'))

def information(data):
    info = 0.0
    uniqueValues = data.value_counts()
    for count in uniqueValues:
        p = probability(count, len(data))
        if p != 0:
            temp1 = p * (log(p, 2))
        else:
            temp1 = 0
        info -= temp1
    return float(format(info, '.5f'))

def probability(s1, s):
    s1 = abs(s1)
    s = abs(s)
    if s1 != 0 and s != 0:
        result = s1 / s
    else:
        result = 0
    return float(format(result, '.5f'))

def classify(data, input):
    if isinstance(data, tuple):
        if data[0] in input:
            attribute_data = input[data[0]]
            if attribute_data in data[1]:
                value = data[1].get(attribute_data)
                result = classify(value, input)
            else:
                value = data[1].get('default')
                result = classify(value, input)
        else:
            value = data[1].get('default')
            result = classify(value, input)
    else:
        result = data
    print("Result: ", result)
    return result

def dataPreprocesing(data, label):
    dicList = list()
    for row in data:
        temp = {label: row[1]}
        row[0].update(temp)
        dicList.append(row[0])
    dataFrame = pa.DataFrame(data=dicList)
    return dataFrame

training_data = [
    ({'a1': 'True', 'a2': 'Hot', 'a3': 'High'}, "No"),
    ({'a1': 'True', 'a2': 'Hot', 'a3': 'High'}, "No"),
    ({'a1': 'False', 'a2': 'Hot', 'a3': 'High'}, "Yes"),
    ({'a1': 'False', 'a2': 'Cool', 'a3': 'Normal'}, "Yes"),
    ({'a1': 'False', 'a2': 'Cool', 'a3': 'Normal'}, "Yes"),
    ({'a1': 'True', 'a2': 'Cool', 'a3': 'High'}, "No"),
    ({'a1': 'True', 'a2': 'Hot', 'a3': 'High'}, "No"),
    ({'a1': 'True', 'a2': 'Hot', 'a3': 'Normal'}, "Yes"),
    ({'a1': 'False', 'a2': 'Cool', 'a3': 'Normal'}, "Yes"),
    ({'a1': 'False', 'a2': 'Cool', 'a3': 'High'}, "Yes")
]

label = 'Result'
dataFrame = dataPreprocesing(training_data, label)
dt = ID3(dataFrame, label)

print('\nID3 classification result : \n')
c1 = {"a1": "True", "a2": "Cool", "a3": "Normal"}
c2 = {"a1": "False", "a2": "Hot", "a3": "High"}

print("\nClassify1 = ", c1, '\n')
print("Classify1 Result = ", classify(dt, c1), '\n')
print("Classify2 = ", c2, '\n')
print("Classify2 Result = ", classify(dt, c2), '\n')
