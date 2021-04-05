import pandas as pd
df1 = pd.DataFrame({'lkey1': ['foo', 'bar', 'baz', 'foo'],
                    'value1': [1, 2, 3, 5],
                   'letter':["A","N","C","D"]})

df2 = pd.DataFrame({'rkey2': ['foo', 'bar', 'baz', 'foo'],
                    'value2': [5, 6, 7, 5],
                    'letter':["A","B","C","N"]})

df = pd.merge(df1,df2,on='letter',how="outer")

res = df.loc[(df.value1==5)|(df.value2==5)]
print(res)
print((0.003*0.99)/(0.003*0.99+0.997*0.001))