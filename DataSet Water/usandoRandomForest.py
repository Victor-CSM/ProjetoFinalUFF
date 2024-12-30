import pandas as pd

# Carregar o dataset
file_path = 'water_potability_cleaned.csv'
data = pd.read_csv(file_path)



#Fazer aquela normalização
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical_columns = data.columns[:-1]  # Todas as colunas, exceto 'Potability'
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# print(data.head()) #Mostra as 5 primeiras linhas
# print(data[numerical_columns].min())
# print(data[numerical_columns].max())

#Seção de divisão do dataSet em treino e teste
from sklearn.model_selection import train_test_split

# Separar variáveis preditoras (X) e alvo (y)
X = data.drop('Potability', axis=1)  # Todas as colunas, exceto 'Potability'
y = data['Potability']  # Apenas a coluna 'Potability'

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n-------------------------------------------------------------- RandomForest --------------------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



# Ajustar pesos para balancear as classes
model_rf_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
model_rf_balanced.fit(X_train, y_train)

y_pred_balanced = model_rf_balanced.predict(X_test)

print("Acurácia (com balanceamento):", accuracy_score(y_test, y_pred_balanced))
print(classification_report(y_test, y_pred_balanced))

model_rf_tuned = RandomForestClassifier(
    random_state=42, 
    n_estimators=200,  # Aumentar o número de árvores
    max_depth=10,      # Limitar a profundidade para evitar overfitting
    class_weight='balanced'
)
model_rf_tuned.fit(X_train, y_train)

y_pred_tuned = model_rf_tuned.predict(X_test)

print("Acurácia (com ajuste de hiperparâmetros):", accuracy_score(y_test, y_pred_tuned))
print(classification_report(y_test, y_pred_tuned))



print(data['Potability'].value_counts(normalize=True))
