import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, roc_auc_score
)
import joblib

# ✅ 1. Chargement des données
df = pd.read_csv("data/data.csv")

# ✅ 2. Nettoyage : Remplacement des zéros
cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_replace:
    df[col] = df[col].replace(0, df[col].median())

# ✅ 3. Séparation des variables
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# ✅ 4. Normalisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")
# ✅ 5. Split des données
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ================================
# 🔹 Modèle 1 : Random Forest
# ================================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# 🔹 Optimisation (facultative)
grid_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(RandomForestClassifier(random_state=42), grid_params, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

# ✅ Sauvegarde
joblib.dump(best_rf, "modele_random_forest.pkl")

# ================================
# 🔹 Modèle 2 : Régression Logistique
# ================================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

# ✅ Sauvegarde
joblib.dump(log_model, "modele_logistic_regression.pkl")

# ================================
# 🔍 Comparaison des deux modèles
# ================================
models = {
    "Random Forest (Optimisé)": (y_pred_best_rf, best_rf.predict_proba(X_test)[:, 1]),
    "Régression Logistique": (y_pred_log, log_model.predict_proba(X_test)[:, 1])
}
for name, (y_pred, y_probs) in models.items():
    print(f"\n====== {name} ======")
    print("✅ Accuracy :", accuracy_score(y_test, y_pred))
    print("✅ Rapport de classification:\n", classification_report(y_test, y_pred))
    # Matrice de confusion
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non Diabétique", "Diabétique"],
                yticklabels=["Non Diabétique", "Diabétique"])
    plt.title(f"Matrice de Confusion - {name}")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")
    plt.tight_layout()
    plt.show()
    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc = roc_auc_score(y_test, y_probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("Comparaison des courbes ROC")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.legend()
plt.show()

# ✅ Export vers CSV
results = pd.DataFrame({
    "Réel": y_test,
    "Prédiction_RF": y_pred_best_rf,
    "Prédiction_Log": y_pred_log
})
results.to_csv("resultats_comparaison.csv", index=False)


# ========================
# 🔥 Tester avec une nouvelle personne
# ========================
import numpy as np

modele = joblib.load("modele_logistic_regression.pkl")
scaler = joblib.load("scaler.pkl")

nouvelle_personne = np.array([[2, 130, 80, 25, 90, 32.0, 0.45, 29]])
nouvelle_personne_scaled = scaler.transform(nouvelle_personne)
prediction = modele.predict(nouvelle_personne_scaled)

if prediction[0] == 1:
    print("✅le patient est diabetique. ")
else:
    print("✅le patient n'est pas diabetique.")

