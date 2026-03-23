import numpy
import rpy2.robjects as r
import rpy2.robjects.numpy2ri
from sklearn.metrics import mean_squared_error
rpy2.robjects.numpy2ri.activate()
import DataHandler as dh
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

class Panigrahi:
    def __init__(self,data,dimension,neurons,testNO):
        self.data=data
        self.dimension=dimension
        self.trainset=0.8
        self.valset=0.2
        self.neurons=neurons
        self.ndata=(data-min(data))/(max(data)-min(data))
        self.testNO=testNO; #lynx
    def start(self):

        dh2=dh.DataHandler(self.ndata,self.dimension,self.trainset,self.valset,self.testNO)
        train_set, train_target, val_set, val_target, test_set, test_target, arima_train, arima_val, arima_test= dh2.redimensiondata(self.ndata,self.dimension,self.trainset,self.valset,self.testNO)
        traindats=[]
        traindats.extend(arima_train)
        traindats.extend(arima_val)

        r.r('library(forecast)')
        ets = r.r('ets')
        arima_train.extend(arima_val)
        numeric = r.r('as.numeric')
        fit = ets(numeric(arima_train))
        fitted = r.r('fitted')
        predTreino = fitted(fit)
        fit3 = ets(numeric(arima_test),'use.initial.values=TRUE', model=fit)

        predTest = fitted(fit3)

        predTudo=[]
        predTudo.extend(predTreino)
       # target=[]
        #target.extend(arima_train)
        #arget.extend(arima_val)
        
        predTudo.extend(predTest)
        residual=self.ndata-predTudo

        residualNorm= (residual-min(residual))/(max(residual)-min(residual))

        train_set2, train_target2, val_set2, val_target2, test_set2, test_target2, arima_train2, arima_val2, arima_test2 = dh2.redimensiondata(
            residualNorm, self.dimension, self.trainset, self.valset,self.testNO)
        train_set2.extend(val_set2)
        train_target2.extend(val_target2)
        rna2= MLPRegressor(hidden_layer_sizes=(self.neurons,),activation='logistic',solver='lbfgs',shuffle=False)
        rna = GridSearchCV(rna2, param_grid={
            'hidden_layer_sizes': [(2,), (5,), (10,), (15,), (20,)]})
        rna.fit(train_set2,train_target2)
        predRNA=rna.predict(test_set2)
        predRNAD=predRNA*(max(residual)-min(residual))+min(residual)

        predFinalN=numpy.asarray(predTest)+numpy.asarray(predRNAD)


       
        mse=mean_squared_error(arima_test,predFinalN)

        arima_testdn=numpy.array(arima_test)*(max(self.data)-min(self.data))+min(self.data)
        preddn=numpy.array(predFinalN)*(max(self.data)-min(self.data))+min(self.data)
        #msednorm=mean_squared_error(arima_testdn,preddn)
        mape = numpy.mean(numpy.abs((arima_testdn - preddn)/arima_testdn))
        print('MSE = %f'%mse)
        print('MAPE = %f'%mape)
        return mse,predFinalN
#Star	Paper	Nordic	Milk	Lake	Gas  	B1H 	Colorado	Airline	CarSales
#24     12       24      24      12      12     24       12          12     12    
#120     24       600    30      100     14     300      150         24     20

x=pd.read_excel('Maceio.xlsx')
x.iloc[:,6].fillna(0)

dataset = x.iloc[:,6].fillna(0).values
zg=Panigrahi(dataset, 24, 20, 299)
(mseval,predFinalN)=zg.start()
