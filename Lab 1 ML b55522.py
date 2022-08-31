import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df  = pd.read_csv("titanic.csv")

#queremos remover name, ticket,passengerID y cabin porque no parecen 
#importantes a la hora
#de analisar, tambien se podriar remover embark dado que su informacion no 
#parece importante

df_ct = df.loc[:,['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]

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
        self.inercia = 1/len(self.matrix[0])*( self.eigenVal* self.c )
        return self.inercia
        
pca = myPCA(matrix,std_dev,means)

pca.matrix_cent_redux()
pca.matrix_correlation()
pca.eigen()
pca.calc_c()
pca.inercia()
