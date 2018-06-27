import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import pickle
import os
import numpy as np


path = '../dataset/gat2017log15_dataset.pickle'
current_path = os.path.dirname(os.path.abspath(__name__))
joined_path = os.path.join(current_path, path)
data_path = os.path.normpath(joined_path)

# データ読み込み
with open(data_path, mode='rb') as f:
    dataset = pickle.load(f)

train = dataset['train']
test = dataset['test']

train_data = [[],[]]
test_data = [[],[]]

# 0日目は除く
for i in range(len(train)):
    if(train[i][0][10] != 0):
        train_data[0].append(train[i][0])
        train_data[1].append(train[i][1])
print("train:"+str(len(train)))
for i in range(len(test)):
    if(test[i][0][10] != 0):
        test_data[0].append(test[i][0])
        test_data[1].append(test[i][1])
print("test:"+str(len(test)))

train_data[0] = np.array(train_data[0], dtype=np.int32)
train_data[1] = np.array(train_data[1], dtype=np.int32)
test_data[0] = np.array(test_data[0], dtype=np.int32)
test_data[1] = np.array(test_data[1], dtype=np.int32)

# xgboostモデルの作成
clf = xgb.XGBClassifier()

# ハイパーパラメータ探索
clf_cv = GridSearchCV(clf, {'max_depth': [2,4,6], 'n_estimators': [5, 10, 20, 50, 100]}, verbose=1)
clf_cv.fit(train_data[0], train_data[1])
print(clf_cv.best_params_, clf_cv.best_score_)

# 改めて最適パラメータで学習
clf = xgb.XGBClassifier(**clf_cv.best_params_)
clf.fit(train_data[0], train_data[1])

# 学習モデルの保存、読み込み
# import pickle
# pickle.dump(clf, open("model.pkl", "wb"))
# clf = pickle.load(open("model.pkl", "rb"))

# 学習モデルの評価
pred = clf.predict(test_data[0])
print(confusion_matrix(test_data[1], pred))
print(classification_report(test_data[1], pred))
