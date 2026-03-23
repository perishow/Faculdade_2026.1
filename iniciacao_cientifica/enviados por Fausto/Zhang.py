import pandas as pd
import numpy
import os
os.environ['R_HOME'] = '/usr/lib/R'
import rpy2.robjects as r
import rpy2.robjects.numpy2ri
from sklearn.metrics import mean_squared_error
rpy2.robjects.numpy2ri.activate()
import DataHandler as dh
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")
class Zhang:
    def __init__(self,data,dimension,neurons,testNO):
	    # data = dados que serão trabahados
		# dimension = dimensão dos dados, (janela)
		# neurons = número de neurônios escondidos
		# testNO = número de instâncias de teste
        self.data=(data -min(data))/(max(data)-min(data))
        self.dimension=dimension
        self.trainset=0.8
        self.valset=0.2
        self.neurons=neurons
        self.testNO=testNO
    def start(self):
        dh2=dh.DataHandler(self.data,self.dimension,self.trainset,self.valset,self.testNO)
        train_set, train_target, val_set, val_target, test_set, test_target, arima_train, arima_val, arima_test= dh2.redimensiondata(self.data,self.dimension,self.trainset,self.valset,self.testNO)
        traindats=[]
        traindats.extend(arima_train)
        traindats.extend(arima_val)

        r.r('library(forecast)')
        arima = r.r('auto.arima')
        arimaTest=r.r('Arima')
        ordem = r.r('c')
        #arima_train.extend(arima_val)
        numeric = r.r('as.numeric')
        fit = arima(numeric(arima_train))
        fitted = r.r('fitted')
        predTreino = fitted(fit)
        fit2 = arimaTest(numeric(arima_val),model=fit)
        fit3 = arimaTest(numeric(arima_test), model=fit)
        predVal = fitted(fit2)
        predTest = fitted(fit3)
        
        predTudo=[]

        predTudo.extend(predTreino)
        predTudo.extend(predVal)
       # target=[]
        #target.extend(arima_train)
        #arget.extend(arima_val)
        residualTreino=numpy.array(arima_train)-(predTreino)
        predTudo.extend(predTest)
        residual=self.data-predTudo

        residualNorm= (residual-min(residualTreino))/(max(residualTreino)-min(residualTreino))
        split = []
        split =[ 'Treino' for x in range(len(self.data)) if x<len(arima_train)  ]
        for x in range(len(arima_val)):
            split.append('Valid')
        for x in range(len(arima_test)):
            split.append('test')
        
        my_ditc = OrderedDict({'Data':self.data,'ARIMA':predTudo,'Residuo':residual,'Split':split})
        
        # ARIMA + MLP
        my_df = pd.DataFrame(my_ditc)
        train_set2, train_target2, val_set2, val_target2, test_set2, test_target2, arima_train2, arima_val2, arima_test2 = dh2.redimensiondata(
            residualNorm, self.dimension, self.trainset, self.valset,self.testNO)
        train_set2.extend(val_set2)
        train_target2.extend(val_target2)
        nn1 = MLPRegressor(activation='relu', solver='lbfgs', shuffle=False)
        rna = GridSearchCV(nn1, param_grid={
            'hidden_layer_sizes': [(2,), (5,), (10,), (15,), (20,)]})
        rna.fit(train_set2,train_target2)
#        print('Number of hidden neurons %d'%(neuronlist[index]))
        predRNA=rna.predict(test_set2)

        predRNAD=predRNA*(max(residualTreino)-min(residualTreino))+min(residualTreino)

        predFinal=numpy.asarray(predTest)+numpy.asarray(predRNAD)
        #print('Results Without normalization:')
        #print('MSE = %f'%mean_squared_error(test_target,predFinal))
        #print('MAPE = %f'%numpy.mean(numpy.abs((test_target-predFinal)/test_target)))
        predFinalN=(numpy.asarray(predFinal)-min(traindats))/(max(traindats)-min(traindats))

        testTarget=(numpy.asarray(arima_test)-min(traindats))/(max(traindats)-min(traindats))
        mse=mean_squared_error(testTarget,predFinalN)
        
        print('MSE ARIMA = %f'%(mean_squared_error(numpy.array(predTest),numpy.array(arima_test))))
        print('MSE ZHANG = %f\n'%(mse))
        
        return mse,predFinalN

x=pd.read_excel('Maceio.xlsx')
x.iloc[:,6].fillna(0)

dataset = x.iloc[:,6].fillna(0).values
zg=Zhang(dataset, 24, 20, 299)
zg.start()
