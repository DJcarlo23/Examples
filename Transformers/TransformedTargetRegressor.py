import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

df = pd.DataFrame(columns=['X1', 'X2', 'y'], data=[
    [1, 16, 81],
    [4, 36, 256],
    [1, 16, 81],
    [2, 9, 64],
    [3, 36, 225],
    [2, 49, 256],
    [4, 25, 196],
    [5, 36, 289]
])

# sqrt(y) = X1 + 2 * sqrt(X2)

train = df.iloc[:6]
test = df.iloc[6:]

train_X = train.drop('y', axis=1)
train_y = train.y

test_X = test.drop('y', axis=1)
test_y = test.y

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

class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    # no need to implement __init__ in this particular case

    def fit(self, target):
        return self

    def transform(self, target):
        print('\n%%%%%%%%%%% custom_target_transform() called.\n')
        target_ = target.copy()
        target_ = np.sqrt(target_)
        return target_

    # need to implement this too
    def inverse_transform(self, target):
        print('\n%%%%%%%%%%%%%%% custom_inverse_target_transform() called.\n')
        target_ = target.copy()
        target_ = target_ ** 2
        return target_

def target_transform(target):
    print('\n***************target_transform() called.\n')
    target_ = target.copy()
    target_ = np.sqrt(target_)
    return target_

def inverse_target_transform(target):
    print('\n***************inverse_target_transform() called.\n')
    target_ = target.copy()
    target_ = target_ ** 2
    return target_

# with input transformation & target transformation
print("create pipeline 3")
# no change in input pipeline
pipe3 = Pipeline(steps=[
    ('experimental_trans', ExperimentalTransformer_2('X2')),
    ('linear_model', LinearRegression())
])

# create a TargetTransformer
model = TransformedTargetRegressor(regressor=pipe3,
                                   func=target_transform,
                                   inverse_func=inverse_target_transform)

print("fit pipeline 3 [fit Model]")
# note the different syndax here; we fit the 'model' now, instead of 'pipe3'
model.fit(train_X, train_y)
print("predict via pipeline 3 [Model]")
preds3 = model.predict(test_X) # same here, using 'model' to predict
print(f"\n{preds3}") # should be [196. 289.]
print(f"RMSE: {np.sqrt(mean_squared_error(test_y, preds3))}\n")

# with input transformation & target transformation
print("create pipeline 3.1")
# no change in input pipeline
pipe3_1 = Pipeline(steps=[
    ('experiment_trans', ExperimentalTransformer_2('X2')),
    ('linear_model', LinearRegression())
])

# create a TargetTransformer
model = TransformedTargetRegressor(regressor=pipe3_1,
                                   transformer=CustomTargetTransformer(),
                                   check_inverse=False) # avoid repeated calls
print("fit pipeline 3.1 [fit Model]")
model.fit(train_X, train_y)
print("predict via pipeline 3.1 [Model]")
preds3_1 = model.predict(test_X)
print(f"\n{preds3_1}") # should be [196. 289.]
print(f"RMSE: {np.sqrt(mean_squared_error(test_y, preds3_1))}\n")