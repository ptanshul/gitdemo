from sklearn.datasets import load_boston
import pandas as pd

boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target
data.to_csv('boston_housing.csv', index=False)
