import kagglehub
import pandas as pd
import os
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Baixar o dataset do Kaggle
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
file_path = os.path.join(path, "creditcard.csv")

# Carregar os dados
df = pd.read_csv(file_path)

# Análise exploratória
print(df.info())
print("\nDistribuição das Classes:")
print(df["Class"].value_counts())

# Gráfico de distribuição
fraud_counts = df["Class"].value_counts()
plt.figure(figsize=(6, 4))
plt.bar(["Não Fraude", "Fraude"], fraud_counts, color=["blue", "red"])
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.title("Distribuição das Classes no Dataset")
plt.show()

# Balanceamento dos dados (SMOTE)
X = df.drop("Class", axis=1)
y = df["Class"]
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinar modelos
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\nRelatório de Classificação - {name}:")
    print(classification_report(y_test, y_pred))
