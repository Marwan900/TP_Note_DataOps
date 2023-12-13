import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Charger le fichier CSV
file_path = 'C:/Users/marwa/Downloads/OneDrive_2023-12-13/TP noté/Data/train.csv'
df = pd.read_csv(file_path)

# Gestion des valeurs manquantes
df['Age'].fillna(df['Age'].median(), inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# Encodage one-hot pour la colonne 'Name'
df = pd.get_dummies(df, columns=['Name'])
# Encodage one-hot pour la colonne 'Sex'
df = pd.get_dummies(df, columns=['Sex'])

# Exporter les données nettoyées
df.to_csv('cleaned_train.csv', index=False)

# Chargement des données nettoyées
data = pd.read_csv('cleaned_train.csv')

# Séparation des caractéristiques et de la cible
X = data.drop('Survived', axis=1)
y = data['Survived']