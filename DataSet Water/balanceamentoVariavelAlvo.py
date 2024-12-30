import pandas as pd

# Carregar o dataset
file_path = 'water_potability.csv'
data = pd.read_csv(file_path)
print(data['Potability'].value_counts(normalize=True))


