import pandas as pd

# Carregar o dataset
file_path = 'water_potability.csv'
data = pd.read_csv(file_path)

# Explorar as primeiras linhas do dataset
data.head(), data.info(), data.describe()

# Excluir linhas com valores ausentes
data.dropna(inplace=True)

# Verificar novamente se ainda há valores ausentes
print(data.isnull().sum())

# Confirmar o número de linhas restantes
print(f"Linhas restantes: {len(data)}")

# Excluir linhas com valores ausentes
data.dropna(inplace=True)

# Salvar o dataset atualizado em um novo arquivo CSV
#data.to_csv('water_potability_cleaned.csv', index=False)
