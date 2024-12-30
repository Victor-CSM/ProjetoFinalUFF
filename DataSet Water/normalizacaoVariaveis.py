import pandas as pd

# Carregar o dataset
file_path = 'water_potability_cleaned.csv'
data = pd.read_csv(file_path)

from sklearn.preprocessing import MinMaxScaler

#Fazer aquela normalização
scaler = MinMaxScaler()
numerical_columns = data.columns[:-1]  # Todas as colunas, exceto 'Potability'
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

print(data.head()) #Mostra as 5 primeiras linhas

print(data[numerical_columns].min())
print(data[numerical_columns].max())


#Seção de divisão do dataSet em treino e teste
from sklearn.model_selection import train_test_split

# Separar variáveis preditoras (X) e alvo (y)
X = data.drop('Potability', axis=1)  # Todas as colunas, exceto 'Potability'
y = data['Potability']  # Apenas a coluna 'Potability'

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar as dimensões dos conjuntos
print(f"Conjunto de treino: {X_train.shape}, {y_train.shape}")
print(f"Conjunto de teste: {X_test.shape}, {y_test.shape}")

print("Primeiras linhas do conjunto de treino:")
print(X_train.head())

print("\nPrimeiras linhas do conjunto de teste:")
print(X_test.head())

print("Distribuição no conjunto de treino:")
print(y_train.value_counts(normalize=True))

print("\nDistribuição no conjunto de teste:")
print(y_test.value_counts(normalize=True))


#Treinar e testar -Árvore de Decisão
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Inicializar o modelo
model = DecisionTreeClassifier(random_state=42)

# Treinar o modelo com os dados de treino
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
print("\n-------------------------------------------------------------- Árvore de Decisão --------------------------------------------------------------")
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("\nAcurácia:")
print(accuracy_score(y_test, y_pred))


# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

# Inicializar o modelo KNN
knn_model = KNeighborsClassifier(n_neighbors=5)  # 5 vizinhos

# Treinar o modelo com os dados de treino
knn_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred_knn = knn_model.predict(X_test)

# Avaliar o modelo
print("\n-------------------------------------------------------------- K-Nearest Neighbors --------------------------------------------------------------")
print("KNN - Relatório de Classificação:")
print(classification_report(y_test, y_pred_knn))

print("\nKNN - Acurácia:")
print(accuracy_score(y_test, y_pred_knn))


# Support Vector Machines (SVM)
from sklearn.svm import SVC

# Inicializar o modelo SVM
svm_model = SVC(kernel='linear', random_state=42)  # Kernel linear

# Treinar o modelo com os dados de treino
svm_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred_svm = svm_model.predict(X_test)

# Avaliar o modelo
print("\n-------------------------------------------------------------- Support Vector Machines --------------------------------------------------------------")
print("SVM - Relatório de Classificação:")
print(classification_report(y_test, y_pred_svm))

print("\nSVM - Acurácia:")
print(accuracy_score(y_test, y_pred_svm))



# Regressão Logística
from sklearn.linear_model import LogisticRegression

# Inicializar o modelo de Regressão Logística
logistic_model = LogisticRegression(random_state=42)

# Treinar o modelo com os dados de treino
logistic_model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred_logistic = logistic_model.predict(X_test)

# Avaliar o modelo
print("\n-------------------------------------------------------------- Regressão Logística --------------------------------------------------------------")
print("Regressão Logística - Relatório de Classificação:")
print(classification_report(y_test, y_pred_logistic))

print("\nRegressão Logística - Acurácia:")
print(accuracy_score(y_test, y_pred_logistic))

#Treinamento com o algoritmo RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n-------------------------------------------------------------- RandomForest --------------------------------------------------------------")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


