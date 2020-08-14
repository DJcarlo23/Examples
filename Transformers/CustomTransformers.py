import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

class ExperimentalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('\n>>>>>>>>init() called.\n')

    def fit(self, X, y = None):
        print('\n>>>>>>>fit() called.\n')
        return self

    def transform(self, X, y = None):
        print('\n>>>>>>>transform() called.\n')
        X_ = X.copy() # creating a copy to avoid changes to original dataset
        X_.X2 = 2 * np.sqrt(X_.X2)
        return X_

class ExperimentalTransformer_2(BaseEstimator, TransformerMixin):
    # add another additional parameter, just for fun, while we are at it
    def __init__(self, feature_name, additional_parem = "Himashu"):
        print('\n>>>>>>>init() called.\n')
        self.feature_name = feature_name
        self.additional_parem = additional_parem

    def fit(self, X, y = None):
        print('\n>>>>>>>>fit() called.\n')
        print(f'\nadditional parem ~~~~~ {self.additional_parem}\n')
        return self

    def transform(self, X, y = None):
        print('\n>>>>>>>transform() called.\n')
        X_ = X.copy()
        X_[self.feature_name] = 2 * np.sqrt(X_[self.feature_name])
        return X_


df = pd.DataFrame(columns=['X1', 'X2', 'y'], data=[
    [1, 16, 9],
    [4, 36, 16],
    [1, 16, 9],
    [2, 9, 8],
    [3, 36, 15],
    [2, 49, 16],
    [4, 25, 14],
    [5, 36, 17]
])

# y = X1 + 2 * sqrt(X2)

train = df.iloc[:6]
test = df.iloc[6:]

train_X = train.drop('y', axis=1)
train_y = train.y

test_X = test.drop('y', axis=1)
test_y = test.y

# let's see if linear regression is able to predict this properly
m1 = LinearRegression()
fit1 = m1.fit(train_X, train_y)
preds = fit1.predict(test_X)
print(f"\n{preds}")
print(f"RMSE: {np.sqrt(mean_squared_error(test_y, preds))}\n")

# what if we square-root X2 and multiply by 2?
train_X.X2 = 2 * np.sqrt(train_X.X2)
test_X.X2 = 2 * np.sqrt(test_X.X2)
print(test_X)
m2 = LinearRegression()
fit2 = m2.fit(train_X, train_y)
preds = fit2.predict(test_X)
print(f"\n{preds}")
print(f"RMSE: {np.sqrt(mean_squared_error(test_y, preds))}\n")

# without input transformation - to validate that we get the same results as before
print("create pipeline 1")
pipe1 = Pipeline(steps=[
    ('experimental_trans', ExperimentalTransformer()),
    ('linear_model', LinearRegression())
])

# an alternate, shorter syntax to do the above, without naming each step, is:
# pipe2 = make_pipeline(ExperimentalTransformer(), LinearRegression())

print("fit pipeline 1")
pipe1.fit(train_X, train_y)
print("predict via pipeline 1")
preds1 = pipe1.predict(test_X)
print(f"\n{preds1}")
print(f"RMSE: {np.sqrt(mean_squared_error(test_y, preds1))}\n")

print("create pipeline 2")
pipe2 = Pipeline(steps=[
    ('experimental_trans', ExperimentalTransformer_2('X2')),
    ('linear_model', LinearRegression())
])
print("fit pipeline 2")
pipe2.fit(train_X, train_y)
print("predict via pipeline 2")
preds2 = pipe2.predict(test_X)
print(f"\n{preds2}")
print(f"RMSE: {np.sqrt(mean_squared_error(test_y, preds2))}\n")
