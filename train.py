import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


# Charger les données nettoyées
data = pd.read_csv('cleaned_train.csv')

# Séparation des caractéristiques et de la cible
X = data.drop('Survived', axis=1)
y = data['Survived']


# Division en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Construction et entraînement du modèle
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Évaluation du modèle sur l'ensemble de test
predictions = model.predict(X_test)

# Évaluation du modèle
predictions = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
# Cross Validation
cv_scores = cross_val_score(model, X, y, cv=5)
average_cv_score = cv_scores.mean()

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Average Cross-Validation Score: {average_cv_score}')
