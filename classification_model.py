
#<-----------------------------  Import File ----------------------------------------------------------->

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#<----------------------------------------------------------------------------------------------------->



def predict_data_and_target_class(data):
    x = data.drop('fraudulent', axis=1)
    y= data['fraudulent']
    return x,y

def split_x_y(x,y):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4)
    return train_x,test_x,train_y,test_y


def Classification_model(model,train_x,test_x,train_y,test_y):
    model.fit(train_x, train_y) # train date


    y_prediction = model.predict(test_x)

    print("")
    print("<---------------------- Predict Result ----------------------------->")
    print("")
    print(y_prediction)
    print("")

    # Accurate Result
    print("<---------------------- Accurate Result ----------------------------->")
    test_acc = accuracy_score(test_y, y_prediction)
    print(test_acc)   # ( Tp + tn ) / total




    # predict Error
    print("")
    matrices = metrics.mean_absolute_error(test_y, y_prediction)  # 1 - test_acc = 1 - 0.95 = 0.04
    print("<---------------------- prediction Error --------------------------->")
    print(matrices)
    print("")


    # Confusion matrix
    print("<---------------------- Confusion matrix --------------------------->")
    result = confusion_matrix(test_y, y_prediction).ravel()
    print(result)#[12,34,
                 # 55,67]

    tn, fp, fn, tp = confusion_matrix(test_y, y_prediction).ravel()
    print("tn=", tn)
    print("fp=", fp)
    print("fn=", fn)
    print("tp=", tp)

    if fp == 0 and tp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    sensitivity = tp / (tp + fn)
    specification = tn / (tn + fp)

    print("sensitivity=", sensitivity)
    print("specification", specification)
    print("precision", precision)

    if precision==0.0:
        F1_Score=0.0
    else:
        Recall = tp / (tp + fn)
        F1_Score = 2 * (Recall * precision) / (Recall + precision)

    print("F1_Score=", F1_Score)

    return y_prediction,test_acc,sensitivity,specification,precision,F1_Score,matrices






