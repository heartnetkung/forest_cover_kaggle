# # Boilerplate
get_ipython().run_cell_magic(u'capture', u'', u'%run ../hnk-dskit/boilerplate.ipynb')
get_ipython().system(u' pip install --quiet joblib')
get_ipython().system(u' apt-get update -q')
get_ipython().system(u' apt-get install -q -y --allow-unauthenticated swig')
get_ipython().system(u' pip --quiet install pyrfr')
get_ipython().system(u' pip --quiet install Cython')
get_ipython().system(u' pip --quiet install auto-sklearn')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.special import boxcox1p, inv_boxcox, boxcox
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
import warnings
import autosklearn.classification
from joblib import dump, load
warnings.filterwarnings("ignore")
sns.set()
# # Data Preparation
#! unzip -d data data/all.zip 
get_ipython().system(u' head data/train.csv')
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train.info()
train.head()
test.head()
test['Cover_Type'] = None
raw_df = pd.concat([train,test])
getNA(raw_df)
basicInfo(raw_df)
# ## EDA
heatMap(pd.get_dummies(train, columns=['Cover_Type']).drop(['Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40'],axis=1),abs_cor=0.3 )
heatMap(
  pd.get_dummies(
    train[['Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40','Cover_Type']]
                 , columns=['Cover_Type'])
  ,abs_cor=0.3 )
raw_df.describe()
singleFieldAnalysis('Aspect',raw_df)
singleFieldAnalysis('Slope',raw_df)
singleFieldAnalysis('Elevation',raw_df)
#Horizontal_Distance_To_Hydrology	Vertical_Distance_To_Hydrology	Horizontal_Distance_To_Roadways	Hillshade_9am	Hillshade_Noon	Hillshade_3pm	Horizontal_Distance_To_Fire_Points
singleFieldAnalysis('Hillshade_9am',raw_df)
singleFieldAnalysis('Hillshade_Noon',raw_df)
singleFieldAnalysis('Hillshade_3pm',raw_df)
singleFieldAnalysis('Horizontal_Distance_To_Roadways',raw_df)
singleFieldAnalysis('Horizontal_Distance_To_Hydrology',raw_df)
singleFieldAnalysis('Vertical_Distance_To_Hydrology',raw_df)
# # Feature Engineering
# dump(raw_df, 'raw_df_feat_eng.joblib')
if not 'raw_df' in locals():
  raw_df = load('raw_df_feat_eng.joblib')
raw_df.head()
# ## CV Check
def doSGDClassifierCV(feature, label, metrics="accuracy", cv_folds = 5):
  kf = KFold(
        cv_folds, shuffle=True, random_state=42
    ).get_n_splits(feature.values)
  score = cross_val_score(
      GradientBoostingClassifier(), feature.values, label.values, scoring=metrics, cv=kf
  )
  print("Score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def doCV(df):
  train = df.dropna()
  doSGDClassifierCV(train.drop('Cover_Type',axis=1), train[['Cover_Type']].astype('int8'), metrics="accuracy", cv_folds = 5)
get_ipython().run_cell_magic(u'time', u'', u'doCV(raw_df)')
def doFeatureImportance(df):
  train = df.dropna()
  model = GradientBoostingClassifier()
  feature = train.drop('Cover_Type',axis=1)
  model.fit(feature, train[['Cover_Type']].astype('int8'))
  coef = pd.Series(model.feature_importances_, index=feature.columns)
  print('Removed features:', coef[coef == 0].index)
  coef[coef != 0].sort_values().plot(kind="barh", figsize=(15, 12))
  plt.title("Coefficients in the Lasso Model")
doFeatureImportance(raw_df)
# ## Aspect
raw_df['aspect2'] = np.where(raw_df['Aspect']>180,raw_df['Aspect']-180,raw_df['Aspect']+180 )
# ## Absolute Distance
raw_df['d_hydrology'] = np.sqrt(np.square(raw_df['Horizontal_Distance_To_Hydrology'])+np.square(raw_df['Vertical_Distance_To_Hydrology']) ) 		
# ## Horizontal Distance Interaction
raw_df['d_hydrology_roadways1'] = raw_df['Horizontal_Distance_To_Hydrology'] + raw_df['Horizontal_Distance_To_Roadways']
raw_df['d_hydrology_roadways2'] = np.abs(raw_df['Horizontal_Distance_To_Hydrology'] - raw_df['Horizontal_Distance_To_Roadways'])
raw_df['d_hydrology_fire1'] = raw_df['Horizontal_Distance_To_Hydrology'] + raw_df['Horizontal_Distance_To_Fire_Points']
raw_df['d_hydrology_fire2'] = np.abs(raw_df['Horizontal_Distance_To_Hydrology'] - raw_df['Horizontal_Distance_To_Fire_Points'])
raw_df['d_roadways_fire1'] = raw_df['Horizontal_Distance_To_Roadways'] + raw_df['Horizontal_Distance_To_Fire_Points']
raw_df['d_roadways_fire2'] = np.abs(raw_df['Horizontal_Distance_To_Roadways'] - raw_df['Horizontal_Distance_To_Fire_Points'])
# ## Vertical Distance Interaction
raw_df['d_hydrology_elevation'] = raw_df['Vertical_Distance_To_Hydrology'] + raw_df['Elevation']
raw_df['d_hydrology_elevation2'] = np.abs(raw_df['Vertical_Distance_To_Hydrology'] - raw_df['Elevation'])
# ## CV Check2
get_ipython().run_cell_magic(u'time', u'', u'doCV(raw_df)')
# # Modeling
# dump(raw_df, 'raw_df_model.joblib')
if not 'raw_df' in locals():
  raw_df = load('raw_df_model.joblib')
raw_df.head()
wilderness_columns = ['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4']
soil_columns = ['Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4','Soil_Type5','Soil_Type6','Soil_Type7','Soil_Type8','Soil_Type9','Soil_Type10','Soil_Type11','Soil_Type12','Soil_Type13','Soil_Type14','Soil_Type15','Soil_Type16','Soil_Type17','Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23','Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29','Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35','Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']
raw_df['Wilderness'] = raw_df[wilderness_columns].idxmax(axis=1).str.slice(15).astype('int8')
raw_df['Soil_Type'] = raw_df[soil_columns].idxmax(axis=1).str.slice(9).astype('int8')
raw_df.drop(wilderness_columns, axis=1, inplace=True)
raw_df.drop(soil_columns, axis=1, inplace=True)
raw_df.head()
def model1():
    train = raw_df.dropna()
    feature = train.drop(['Id', 'Cover_Type'],axis = 1)
    label = train['Cover_Type'].astype('int8')
    feature_types = (['numerical']*20)+ (['categorical']*2)
    automl = autosklearn.classification.AutoSklearnClassifier(
        time_left_for_this_task=3600*10,
        per_run_time_limit=3600,
        ml_memory_limit = 20000,
        ensemble_memory_limit=20000,
        output_folder='output',
        tmp_folder='temp',
        delete_output_folder_after_terminate=False,
        delete_tmp_folder_after_terminate=False,
        resampling_strategy = 'cv',
        resampling_strategy_arguments= {'folds':3},
        include_estimators= ['random_forest','extra_trees','k_nearest_neighbors','adaboost','gradient_boosting', 'libsvm_svc'],
        exclude_preprocessors = ['imputation']
    )
    automl.fit(feature, label, dataset_name='forest_cover_type',
               feat_type=feature_types)
    print(automl.show_models())
    dump(automl, 'model_cv.joblib')
get_ipython().run_cell_magic(u'time', u'', u'model1()')
# ## Retrain with all data
final_model = load('model_cv.joblib')
def retrain(model):
    train = raw_df.dropna()
    feature = train.drop(['Id', 'Cover_Type'],axis = 1)
    label = train['Cover_Type'].astype('int8')
    return model.refit(feature,label)
get_ipython().run_cell_magic(u'time', u'', u'final_model2 = retrain(final_model)')
dump(final_model2, 'model_cv2.joblib')
# # Finish Up
final_model2 = load('model_cv2.joblib')
def predict(model, raw_df):
  test_feature = raw_df[raw_df['Cover_Type'].isna()]
  ans=None
  for i in range(0,math.ceil(test_feature.shape[0]/10000)):
    print(i)
    temp = model.predict(test_feature.drop(['Id', 'Cover_Type'],axis = 1).iloc[i*10000:(i*10000)+10000])
    if ans is None:
      ans = temp
    else:
      ans = np.concatenate([ans,temp])
  return ans
get_ipython().run_cell_magic(u'time', u'', u'ans = predict(final_model2, raw_df)')
test_feature = raw_df[raw_df['Cover_Type'].isna()]
test_feature['Cover_Type'] = ans
test_feature[['Id','Cover_Type']].to_csv('prediction3.csv', index=False)
get_ipython().system(u' head prediction3.csv')
final_model2.get_models_with_weights()
print(final_model2.sprint_statistics())
