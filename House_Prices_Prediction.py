from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor,VotingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from helpers.data_prep import *
from helpers.eda import *
import datetime as dt
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

train = pd.read_csv("datasets/house_price/train.csv")
test = pd.read_csv("datasets/house_price/test.csv")

df = train.append(test).reset_index(drop=True)

check_df(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df)


for col in cat_cols:
    cat_summary(df,col)  #Street, utilities, PoolQC, MiscFeature bilgi taşımıyor

for col in num_cols:
    num_summary(df,col) #MasVnrArea, GarageYrBlt


### Eksik Değerlerin Doldurulması ###

missing_values_table(df)

none_cols = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
             'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType']
zero_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt',
             'GarageArea', 'GarageCars', 'MasVnrArea']
freq_cols = ['Exterior1st', 'Exterior2nd', 'KitchenQual']

for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True)
for col in none_cols:
    df[col].replace(np.nan, "None", inplace=True)
for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True)

drop_list = ["Street", "Utilities","PoolQC","MiscFeature","Alley","Fence"]
df.drop(drop_list,axis=1,inplace=True)

missing_values_table(df)

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))

df['LotFrontage'] = df.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

na_cols = [col for col in df.columns if df[col].isnull().sum() > 0 and "SalePrice" not in col]

df[na_cols] = df[na_cols].apply(lambda x: x.fillna(x.mode()[0]), axis=0)

missing_values_table(df) # Bağımlı Değişkendeki eksikler dısındaki tüm eksik veriler dolduruldu

### Feature Engineering ###

def find_correlation(dataframe, numeric_cols, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in numeric_cols:
        if col == "SalePrice":
            pass
        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations

low_correlations, high_correlations = find_correlation(df,num_cols)

df["NEW_TotalAreaSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

df["NEW_LIV_AREA_RATIO"] = df["GrLivArea"] / df["LotArea"]

df["NEW_TotalAreaSF_W_Garage"] = df["NEW_TotalAreaSF"] + df["GarageArea"]

df["NEW_HAVE_GARAGE"] =  (df["GarageYrBlt"] > 0).astype("int")

df["NEW_HAS_POOL"] = (df['PoolArea'] > 0).astype('int')

df['NEW_total_porch_area'] = df['WoodDeckSF'] + df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

df['NEW_Total_Garden'] = df['LotArea'] - df['TotalBsmtSF'] - df['GarageArea'] - df['NEW_total_porch_area'] - df['PoolArea']

df['NEW_TOTAL_AREA/GARDEN'] = df["NEW_TotalAreaSF"] / df['NEW_Total_Garden']

df['NEW_HAVE_2nd_FLOOR'] = (df['2ndFlrSF'] > 0).astype('int')

# Create Neighborhood Cluster
ngb = df.groupby("Neighborhood").SalePrice.mean().reset_index()
ngb["NEW_CLUSTER_NEIGHBORHOOD"] = pd.cut(df.groupby("Neighborhood").SalePrice.mean().values, 4, labels=range(1, 5))
df = pd.merge(df, ngb.drop(["SalePrice"], axis=1), how="left", on="Neighborhood")

ext_map = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}
df['LotShape'] = df['LotShape'].map(ext_map).astype('int')

ext_map = {'Ex': 4, 'Gd': 3, 'TA': 2, 'Fa': 1, 'Po':0}
df['ExterQual'] = df['ExterQual'].map(ext_map).astype('int')
df['ExterCond'] = df['ExterCond'].map(ext_map).astype('int')

df["NEW_EX_QUAL_COND"] = df['ExterQual'] + df['ExterCond']

df["NEW_QUAL_COND"] = df['OverallQual'] + df['OverallCond']

heat_map = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
df['HeatingQC'] = df['HeatingQC'].map(heat_map).astype('Int64')

df["NEW_TotalBath"] =  df['FullBath'] + df['HalfBath'] * 0.5
df['NEW_TOTAL_BATHROOMS'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

df["NEW_HOME_QUALITY_PERC"] =  df["OverallCond"] / df["OverallQual"]


#Rare Encoding

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)

df = rare_encoder(df,cat_cols, 0.01)

rare_analyser(df, "SalePrice", cat_cols)

useless_cols = [col for col in cat_cols if df[col].nunique() == 1 or
                (df[col].nunique() == 2 and (df[col].value_counts() / len(df) <= 0.01).any(axis=None))]
#['NEW_HAS_POOL']

cat_cols = [col for col in cat_cols if col not in useless_cols]

for col in useless_cols:
    df.drop(col, axis=1, inplace=True)

rare_analyser(df, "SalePrice", cat_cols)

# Label Encoding & One-Hot Encoding

cat_cols = cat_cols + cat_but_car

df = one_hot_encoder(df, cat_cols, drop_first=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

rare_analyser(df, "SalePrice", cat_cols)


useless_cols_new = [col for col in cat_cols if (df[col].value_counts() / len(df) <= 0.01).any(axis=None)]

df.drop(useless_cols_new,axis=1,inplace=True)

#Outliers

for col in num_cols:
    print(col, check_outlier(df, col, q1=0.01, q3=0.99))

for col in num_cols:
    replace_with_thresholds(df,col,q1=0.01, q3=0.99)

check_df(df)

######################################
# Modeling
######################################

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()].drop("SalePrice", axis=1)

# y = train_df["SalePrice"]
y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)

##################
# Base Models
##################

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


##################
# Hyperparameter Optimization
##################

## LightGBM

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
               "n_estimators": [500,1000, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}


lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model_lgbm = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model_lgbm, X, y, cv=10, scoring="neg_mean_squared_error")))

# rmse = 0.12209617316357457

## CatBoost

catboost_model = CatBoostRegressor(verbose=False,random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(catboost_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))

catboost_params = {"iterations": [200,500,800],
                   "learning_rate": [0.01,0.05, 0.1],
                   "depth": [3,6,9]}

catboost_gs_best = GridSearchCV(catboost_model,
                            catboost_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model_catboost = catboost_model.set_params(**catboost_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model_catboost, X, y, cv=10, scoring="neg_mean_squared_error")))

# rmse = 0.11854308213717754

## GBM

gbm_model = GradientBoostingRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(gbm_model,
                                        X, y, cv=5, scoring="neg_mean_squared_error")))

gbm_params = {"learning_rate": [0.01,0.05,0.1],
                 "max_depth": [3,5,8],
                 "n_estimators": [500,1000,1500],
                "subsample": [1, 0.5, 0.7]}

gbm_gs_best = GridSearchCV(gbm_model,
                            gbm_params,
                            cv=5,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model_gbm = gbm_model.set_params(**gbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model_gbm, X, y, cv=10, scoring="neg_mean_squared_error")))

# rmse = 0.1184363744190919

#######################################
## Voting Regressor
#######################################

voting_reg = VotingRegressor(estimators=[("GBM",final_model_gbm),
                                         ("CB",final_model_catboost),]).fit(X,y)

rmse_voting = np.mean(np.sqrt(-cross_val_score(voting_reg, X, y, cv=10, scoring="neg_mean_squared_error")))

# final_rmse = 0.11680229811213072




