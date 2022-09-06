import numpy as np
import math

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
        self.corrMatrix = 1/len(self.matrix) *( np.dot( self.matrix.transpose() , self.matrix))
        return self.corrMatrix
    
    #Valores y vectores propios de R
    def eigen(self):
        eigenVal,eigenVec = np.linalg.eig(self.corrMatrix)
        idx = np.argsort(abs(eigenVal))[::-1]
        eigenVal = eigenVal[idx]
        eigenVec = eigenVec[:,idx]
        self.eigenVal = eigenVal
        self.eigenVec = eigenVec 
        return self.eigenVal,self.eigenVec
        
    #C = X * V
    def calc_c(self):
        self.c = np.dot(self.matrix,self.eigenVec)
        return self.c
    # Inercia = λn / m
    def inercia(self):
        self.inercia = 1/len(self.matrix[0])*(self.eigenVal)
        return self.inercia
    
    def graphPoints(self):
        self.points = []
        for i in range(len(self.eigenVec)):
            x = self.eigenVec[0][i]*math.sqrt(abs(self.eigenVal[i]))
            y = self.eigenVec[1][i]*math.sqrt(abs(self.eigenVal[i]))
            self.points.append((x,y))
        self.points = np.matrix(self.points)

    def doPCA(self,pca):
        pca.matrix_cent_redux()
        pca.matrix_correlation()
        pca.eigen()
        pca.calc_c()
        pca.inercia()
        pca.graphPoints()