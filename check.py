import numpy as np
import pandas as pd


data = {
    'Color': ['Red','Red','Red','Yellow','Yellow','Yellow','Yellow','Yellow','Red','Red'],
    'Type': ['Sports','Sports','Sports','Sports','Sports','SUV','SUV','SUV','SUV','SUV']
}

df = pd.DataFrame(data)
df['Color'] = df['Color'].astype(str)

print(df.dtypes)