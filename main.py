
#Ruan Coelho de Lima

import pandas as pd
import matplotlib.pyplot as plt

path = "student_scores.csv"
data = pd.read_csv(path)

#print(data.head())
#print(data.shape)

def graficador():
  fig, ax = plt.subplots(figsize=(10,6))
  ax.scatter(x = data["Hours"], y = data["Scores"])
  plt.xlabel("Horas")
  plt.ylabel("Notas")
  plt.show()

#graficador()


y = data["Scores"].values.reshape(-1,1)
x = data["Hours"].values.reshape(-1,1)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x, y)


#print(4*regressor.coef_ + regressor.intercept_)

#previsão sklearn
#print(regressor.predict([[5]]))

# Para a Validação cruzada
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size = 0.2, random_state = 42)
regressor.fit(x_train, y_train)
preds = regressor.predict(x_test)

# Para a Visualização de resultados
erro = y_test - preds
df_preds = pd.DataFrame({"Actual":y_test.squeeze(), "Previsto":preds.squeeze(), "Erro":erro.squeeze()})
print(df_preds)

# Para a Medição do erro
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rms = mean_squared_error(y_test, preds, squared = False)

print("Mean ab", mae)
print("Mean sq", mse)
print("Mean rms", rms)

# Para o score
print("Score", regressor.score(x_train, y_train))