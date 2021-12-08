# import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()
print("Data keys: ",data.keys())

# Check the output classes
print("target names: ",data['target_names'])

# Check the input attributes
print("feature names: ",data['feature_names'])
# Check the values of eigen vectors
# prodeced by principal components

df1 = pd.DataFrame(data['data'],columns=data['feature_names'])
scaling = StandardScaler()
scaling.fit(df1)
scaled_data = scaling.transform(df1)

principal = PCA(n_components=3)
principal.fit(scaled_data)
x = principal.transform(scaled_data)

print(x.shape)
print("principal components: ",principal.components_)
