import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


df  = pd.read_csv("titanic.csv")

#queremos remover name, ticket,passengerID y cabin porque no parecen 
#importantes a la hora
#de analisar, tambien se podriar remover embark SibSp','Parch' dado que su informacion no 
#parece importante

df_ct = df.loc[:,['Survived','Pclass','Sex','Age','Fare']]

#se limpian los NaN
df_cln = df_ct.dropna()

#se hace el one-hot encoding

df_ready = pd.get_dummies(df_cln,columns=['Sex'])

#se saca la media de las columnas
means = df_ready.mean(0).to_numpy()

#se saca la desviacion estandar de las columnas
std_dev = df_ready.std(0).to_numpy()

#creo la matriz de numpy
matrix = df_ready.to_numpy()


class myPCA:
    #la clase recibe matriz, desviacion estandar y media
    def __init__(self,matrix,std,means):
        self.matrix = matrix
        self.std = std
        self.means = means
        
    #∀xi,j : xi,j’ = (xi,j - μj)/σj
    def matrix_cent_redux(self):
        for i in range(len(self.matrix)):
            for j in range(len(self.matrix[i])):
                self.matrix[i][j] = (self.matrix[i][j] - self.means[j])/self.std[j]
        return self.matrix
        
    #crea matriz de correlacion 1/N*Xt*X = R
    def matrix_correlation(self):
        self.corrMatrix = 1/len(self.matrix) *( np.matmul( self.matrix.transpose() , self.matrix))
        return self.corrMatrix
    
    #Valores y vectores propios de R
    def eigen(self):
        eigenVal,eigenVec = np.linalg.eig(self.corrMatrix)
        idx = eigenVal.argsort()[::-1]
        eigenVal = eigenVal[idx]
        eigenVec = eigenVec[:,idx]
        self.eigenVal = eigenVal
        self.eigenVec = eigenVec
        return self.eigenVal,self.eigenVec
        
    #C = X * V
    def calc_c(self):
        self.c = np.matmul(self.matrix,self.eigenVec)
        return self.c
    # Inercia = λn / m
    def inercia(self):
        self.inercia = 1/len(self.matrix[0])*(self.eigenVal)
        return self.inercia
    
    def graphPoints(self):
        self.points = []
        for i in range(self.eigenVec):
           x = self.eigenVec[0][i]*math.sqrt(testPca.eigenVal[i])
           y = self.eigenVec[1][i]*math.sqrt(testPca.eigenVal[i])
           self.points.append((x,y))
        
def doPCA(pca):
    pca.matrix_cent_redux()
    pca.matrix_correlation()
    pca.eigen()
    pca.calc_c()
    pca.inercia()
    pca.graphPoints()
    
testPca = myPCA(matrix,std_dev,means)

doPCA(testPca)

C = testPca.c
inertia = testPca.inercia
col_corr =testPca.corrMatrix
V = testPca.eigenVec



#plano de PCA
plt.scatter(np.ravel(C[:,0]),np.ravel(C[:,1]),c = ['b' if i==1 else 'r' for i in df_ready['Survived']])
plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[0],))
plt.title('PCA')
plt.show()


#plano de circulo de correlacion
plt.figure(figsize=(15,15))
plt.axhline(0, color='b')
plt.axvline(0, color='b')
for i in range(0, df_ready.shape[1]):
	plt.arrow(0,0, col_corr[i, 0],  # x - PC1
              	col_corr[i, 1],  # y - PC2
              	head_width=0.05, head_length=0.05)
	plt.text(col_corr[i, 0] + 0.05, col_corr[i, 1] + 0.05, df_ready.columns.values[i])
an = np.linspace(0, 2 * np.pi, 100)
plt.plot(np.cos(an), np.sin(an),color="b")  # Circle
plt.axis('equal')
plt.title('Correlation Circle')
plt.show()

#siendo tripulante del titanic da a entender que lo que mas ayudaria a sobrevivir
#es ser mujer


# verificacion con SKLearn

# import sklearn

# df  = pd.read_csv("titanic.csv")

# scaler = sklearn.preprocessing.StandardScaler()
# df_scaled = scaler.fit_transform(df)

# pca = sklearn.decomposition.PCA()
# C = pca.fit_transform(df_scaled)

# inertia = pca.explained_variance_ratio_
# V = pca.transform(np.identity(df_scaled.shape[1]))

# plt.scatter(np.ravel(C[:,0]),np.ravel(C[:,1]),c = ['b' if i==1 else 'r' for i in df['Survived']])
# plt.xlabel('PCA 1 (%.2f%% inertia)' % (inertia[0],))
# plt.ylabel('PCA 2 (%.2f%% inertia)' % (inertia[0],))
# plt.title('PCA')
# plt.show()

# plt.figure(figsize=(15,15))
# plt.axhline(0, color='b')
# plt.axvline(0, color='b')
# for i in range(0, df.shape[1]):
# 	plt.arrow(0,0, col_corr[i, 0],  # x - PC1
#               	col_corr[i, 1],  # y - PC2
#               	head_width=0.05, head_length=0.05)
# 	plt.text(col_corr[i, 0] + 0.05, col_corr[i, 1] + 0.05, df.columns.values[i])
# an = np.linspace(0, 2 * np.pi, 100)
# plt.plot(np.cos(an), np.sin(an),color="b")  # Circle
# plt.axis('equal')
# plt.title('Correlation Circle')
# plt.show()





