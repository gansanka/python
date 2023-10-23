import pandas as pd
import numpy as np

data = {
    "age": [50, 40, 30, 40],
    "qualified": [True, False, False, False],
    "company": ["Game", "sublab", "BC", "Meet"]
}
idx = ["Sally", "Mary", "John", "Monica"]
df = pd.DataFrame(data, index=idx)

print(df)

# reindex
re_idx=["Sally","Mary","R3","R4"]
print(df.reindex(re_idx))

# axis reindexing
axis_rdf = df.reindex(re_idx, axis="index")
print(axis_rdf)

# axis reindexing the rows
axis_rdf = df.reindex(['age','id'], axis="rows", fill_value="missing")
print(axis_rdf)
# axis reindexing the columns
axis_cdf = df.reindex(['age', 'company','id'], axis="columns", fill_value="G8")
print(axis_cdf)

dict_df = pd.DataFrame.from_dict(data)
print(dict_df)

dict_df["leaves"] = dict_df['age'].apply(lambda x: x/10+1)

print(dict_df)

# determinant
arr = np.array([[0,1],[2,3]])
print (np.linalg.det(arr))