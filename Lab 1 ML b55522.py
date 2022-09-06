import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from myPCA import myPCA

df  = pd.read_csv("titanic.csv")

#queremos remover name, ticket,passengerID, embark

df = df.loc[:,['Survived','Pclass','Sex','Age', 'SibSp', 'Parch']]

#se limpian los NaN
df = df.dropna()

#se hace el one-hot encoding

df = pd.get_dummies(df,columns=['Pclass','Sex'])

#se saca la media de las columnas
means = df.mean(0).to_numpy()

#se saca la desviacion estandar de las columnas
std_dev = df.std(0).to_numpy()

#creo la matriz de numpy
matrix = df.to_numpy()

    
testPca =  myPCA(matrix, std_dev, means)

testPca.doPCA(testPca)

centMatrix = testPca.matrix
C = testPca.c
inertia = testPca.inercia
col_corr =testPca.points
V = testPca.eigenVec




#plano de PCA
plt.scatter(np.ravel(C[:,0]),np.ravel(C[:,1]),c = ['b' if i==1 else 'r' for i in df['Survived']])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[1],))
plt.title('PCA')
plt.show()


#plano de circulo de correlacion
plt.figure(figsize=(15,15))
plt.axhline(0, color='b')
plt.axvline(0, color='b')
for i in range(0, df.shape[1]):
	plt.arrow(0,0, col_corr[i, 0],  # x - PC1
                   col_corr[i, 1],  # y - PC2
                   head_width=0.05, head_length=0.05)
	plt.text(col_corr[i, 0] + 0.05, col_corr[i, 1] + 0.05, df.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an),color="b")  # Circle
plt.axis('equal')
plt.title('Correlation Circle')
plt.show()

#siendo tripulante del titanic da a entender que lo que mas ayudaria a sobrevivir
#es ser mujer


# verificacion con SKLearn

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

pca = PCA()
C = pca.fit_transform(df_scaled)

inertia = pca.explained_variance_ratio_
V = pca.transform(np.identity(df_scaled.shape[1]))

print(V)
 

plt.scatter(np.ravel(C[:,0]),np.ravel(C[:,1]),c = ['b' if i==1 else 'r' for i in df['Survived']])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[1],))
plt.title('PCA')
plt.show()

plt.figure(figsize=(15,15))
plt.axhline(0, color='b')
plt.axvline(0, color='b')
for i in range(0, df.shape[1]):
	plt.arrow(0,0, col_corr[i, 0],  # x - PC1
              	col_corr[i, 1],  # y - PC2
              	head_width=0.05, head_length=0.05)
	plt.text(col_corr[i, 0] + 0.05, col_corr[i, 1] + 0.05, df.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an),color="b")  # Circle
plt.axis('equal')
plt.title('Correlation Circle')
plt.show()





