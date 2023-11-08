## we are given a dataset with the following columns (features): how much a company spends on Radio advertising each year and its annual Sales in terms of units sold. 
##We are trying to develop an equation that will let us to predict units sold based on how much a company spends on radio advertising. T
## the rows (observations) represent companies.
import pandas as pd
inp=pd.read_csv('advertising.csv')

def pred_sales(weight,radio,bias):
    return weight*radio+bias

def cost_function(radio,sales,weight,bias):
    cmpn = len(radio)
    total_error = 0.0
    for i in range(cmpn):
        total_error += (sales[i] - (weight*radio[i] + bias))**2
    return total_error / cmpn

def update_weights(radio,sales,weight,bias,learning_rate):
    weight_dev = 0
    bias_dev  = 0
    cmpn = len(radio)
    for i in range(len(radio)):
        # Calculate partial derivatives
        # -2x(y - (mx + b))
        weight_dev += -2 * radio[i] * (sales[i]- (weight*radio[i] + bias))
        #-2(y - (mx + b))
        bias_dev += -2*(sales[i]-(weight*radio[i]+bias))
    weight -= (weight_dev / cmpn)*learning_rate
    bias -= (bias_dev / cmpn)* learning_rate
    
    return weight,bias

def train(radio,sales,weight,bias,learning_rate,iters):
    cost_his = []
    for i in range(iters):
        weight,bias=update_weights(radio,sales,weight,bias,learning_rate)
        cost = cost_function(radio,sales,weight,bias)
        cost_his.append(cost)
        if i %10 == 0:
            print('iter={:d}  weight={:.2f}  bias={:.4f} cost={:.2}'.format(i,weight,bias,cost))
    return weight,bias,cost_his

radio = inp['Radio'].values
sales = inp['Sales'].values
weight = 0
bias = 0
lr = 0.01
iters = 100
train(radio,sales,weight,bias,lr,iters)
  
