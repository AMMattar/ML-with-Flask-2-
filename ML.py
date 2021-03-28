import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('sales.csv')
df.fillna(0, inplace=True)
y = df.iloc[:,-1]
x = df.iloc[:,:3]
def change(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0}
    for i in word:
        if i in word_dict:
            word.replace(i, word_dict[i], inplace=True)

change(x['rate'])

model = LinearRegression()
model.fit(x, y)

pickle.dump(model, open('ml.pkl','wb'))
print(model.predict([5,500,700]))
