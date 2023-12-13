from sklearn.model_selection import train_test_split, cross_val_score 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.metrics import accuracy_score 

import pandas as pd 

 

# Chargement des données nettoyées 

data = pd.read_csv('cleaned_train.csv') 

 

# Séparation des caractéristiques et de la cible 

X = data.drop('Survived', axis=1) 

y = data['Survived'] 

 

# Division en ensembles d'entraînement et de test 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

 

# Construction et entraînement du modèle 

model = RandomForestClassifier() 

model.fit(X_train, y_train) 

 

# Évaluation du modèle 

predictions = model.predict(X_test) 

accuracy = accuracy_score(y_test, predictions) 

print(f'Accuracy: {accuracy}') 

 

# Cross Validation 

scores = cross_val_score(model, X, y, cv=5) 

print(f'Cross-Validation Scores: {scores}') 

print(f'Average Score: {scores.mean()}') 