import pickle
import numpy as np

input = np.array([1.        , 0.        , 1.        , 0.83178337, 0.08421855,
       0.        , 0.        , 1.        , 0.        , 0.        ,
       1.        ])

pickled_model = pickle.load(open('regression.pkl','rb'))
print(pickled_model.predict(input))
