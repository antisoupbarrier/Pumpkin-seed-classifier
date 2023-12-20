import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

df = pd.read_csv('data/Pumpkin_Seeds_Dataset/Pumpkin_Seeds_Dataset.csv')

# parameters

learning_rate = 0.075
n_splits =  10
output_file = f'model_LR={learning_rate}.bin'

# Lowercase column names, spaces replaced with underscore
df.columns = df.columns.str.lower().str.replace(' ', '_')
# Class Mapping of seed name column for training
class_mapping = {'Çerçevelik': 0, 'Ürgüp Sivrisi': 1}
df['class'] = df['class'].map(class_mapping)


X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Save Scaler scales for use by prediction service
filename = "./scaler/"
os.makedirs(os.path.dirname(filename), exist_ok=True)

std  = np.sqrt(scaler.var_)
np.save('scaler/std.npy',std)
np.save('scaler/mean.npy',scaler.mean_)
print(f'Saved StandardScaler variables to scaler/')

# training 

def train(X, y, learning_rate=0.075):
    model = XGBClassifier(learning_rate=learning_rate, n_estimators=100, max_depth=2, verbosity=0, scale_pos_weight=1.083)
    model.fit(X_train, y_train)
    return model

# validation

print(f'doing validation with learning_rate={learning_rate}')


def KFold_Validation_accuracy(model, n_splits):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    print(f'Scores for each fold are: {score}')
    print(f'Average accuracy score: {"{:.3f}".format(score.mean())}')


# training the final model

print('training the final model')

model = train(X_train, y_train, learning_rate=learning_rate)
KFold_Validation_accuracy(model, n_splits)
y_pred = model.predict(X_test)
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((model), f_out)

print(f'the model is saved to {output_file}')