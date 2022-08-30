import pandas as pd
import numpy as np

df  = pd.read_csv("titanic.csv")

#queremos remover name, ticket,passengerID y cabin porque no parecen importantes a la hora
#de analisar, tambien se podriar remover embark dado que su informacion no 
#parece importante

df = df.loc[:,['Survived','Pclass','Sex','Age','SibSp','Parch','Fare']]

#se hace el one-hot encoding

df = pd.get_dummies(df,columns=['Sex'])


df2 = df.dropna()
