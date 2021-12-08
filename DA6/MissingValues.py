import pandas as pd
import numpy as np

dictionary = {'First Score':[87,82,np.nan,96], 'Second Score':[42,32,77,np.nan], 'Third Score':[np.nan,52,97,22]}

df = pd.DataFrame(dictionary)

df = df.fillna(0)

print("Dataframe: ", df)

