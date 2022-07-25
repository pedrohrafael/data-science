import numpy as np

class LogisticRegression:
    """
    Classificador de regressão logística.
    Parametros
    ----------
    n_iterations: int, default=500
        Número máximo de iterações para convergir.
    learning_rate float, default=0.01
        Taxa de aprendizado.
    ----------
    """    
    # Inicializando a função com os parametros learning_rate e n_iterations
    def __init__(self, learning_rate=0.01, n_iterations=500):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    # Implementando a função logistica
    def __sigmoid(self, z):
        np.seterr(all='ignore')
        return 1 / (1 + np.exp(-z))
    
    # Implementando a função de custo: Entropia Cruzada Binária/Log Loss
    def __log_loss(self, y, yhat):        
        return (-1 / self.__m) * np.sum((y.T.dot(np.log(yhat))) + ((1 - y).T.dot(np.log(1 - yhat))))
    
    # Implementando a função de otimização: Gradiente Descendente
    def __gradient_descent(self, X, y, yhat, theta):
        theta -= self.learning_rate * ((1 / self.__m) * (np.dot(X.T, (yhat - y))))
        return theta
    
    # Definindo a função de ajuste do modelo: processo de treinamento
    def fit(self, X, y):
        """
        Ajuste o modelo de acordo com os dados de treinamento fornecidos.
        """
        self.classes_ = np.unique(y)
        self.__m = np.float64(X.shape[0])
        self.loss_lst = list()
        theta = np.zeros((X.shape[1]))
        for _ in range(self.n_iterations):
            z = np.dot(X, theta)            
            yhat = self.__sigmoid(z)
            theta = self.__gradient_descent(X, y, yhat, theta)
            loss = self.__log_loss(y, yhat)
            self.loss_lst.append(loss)
        self.theta = theta        
    
    # Definindo a função de estimador do modelo
    def predict(self, X):
        """
        Prever rótulos de classe para amostras em X.
        """
        z = np.dot(X, self.theta)
        proba = self.__sigmoid(z)
        return np.asarray([1 if p > 0.5 else 0 for p in proba])