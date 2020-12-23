import numpy as np
import csv

def import_data():
    X=np.genfromtxt("train_X_lr.csv",delimiter=",",dtype=np.float64,skip_header=1)
    Y=np.genfromtxt("train_Y_lr.csv",delimiter=",",dtype=np.float64)
    return X,Y
def compute_gradient_of_cost_function(X,Y,W):
    h=np.dot(X,W)
    diff=h-Y
    dw=np.dot(diff.T,X)
    dw=dw.T
    return dw*(1/len(X))
def optimization(X,Y,W,learning_rate):
    previous_iter=0
    iteration_no=0
    while True:
        iteration_no+=1
        dw=compute_gradient_of_cost_function(X,Y,W)
        W=W-(learning_rate*dw)
        cost=cost_function(X,Y,W)
        if iteration_no%1000==0:
            print(iteration_no,cost)
        if abs(previous_iter-cost)<=0.000001:
            print(iteration_no,cost,"end")
            break
        previous_iter=cost
    return W
def cost_function(X,Y,W):
    h=np.dot(X,W)
    diff=h-Y
    sq=np.square(diff)
    mse=np.sum(sq)
    cost_value=mse/(2*len(X))
    return cost_value
def train_model(X,Y):
    X=np.insert(X,0,1,axis=1)
    Y=Y.reshape(len(X),1)
    W=np.zeros((X.shape[1],1))
    W=optimization(X,Y,W,0.0002)
    return W
def save_model(weights,wgt_file):
    with open(wgt_file,"w") as file:
        wr=csv.writer(file)
        wr.writerows(weights)
        file.close()

if __name__=='__main__':
    X,Y=import_data()
    weights=train_model(X,Y)
    save_model(weights,"WEIGHTS_FILE.csv")
