import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("Inc_Exp_Data.csv")
cols = [0, 1, 2]
df = df[df.columns[cols]]
df_filter = (df[df['Mthly_HH_Income'] > df['Mthly_HH_Expense']])

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df_filter.to_numpy())
df_scaled = pd.DataFrame(df_scaled, columns=[
  'Mthly_HH_Income','Mthly_HH_Expense','No_of_Fly_Members'])
print(df_scaled.head())

x = np.linspace(-1,1,10)
y = np.linspace(-1,1,10)
X,Y = np.meshgrid(x,y)
Z=2.29931794e-01*X + 3.36415604e+03*Y -4815.655280927676

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z)
