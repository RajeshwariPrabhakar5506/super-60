import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



#x = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
#print(x.shape)
#y=np.array([1,2,3,4,5,6,7,8,9,10])
#print(y.shape)


x=np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y=np.array([[0], [0], [0], [0], [1], [1], [1], [1], [1], [1]])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#training the model
model=LogisticRegression()
model.fit(x_train,y_train)


#making predictions
y_pred=model.predict(x_test)
print("predictions:", y_pred)
print("actual:", y_test)
accuracy=accuracy_score(y_test,y_pred)

#printing the accuracy
print("Accuracy:", accuracy)


hours=np.array([[7.5],[3.5],[1],[6]])
results=model.predict(hours)
for h,r in zip(hours, results):
    print(f"Hours studied: {h}, Predicted outcome: {'Pass' if r==1 else 'Fail'}") 


#print("if u study for 7.5 hours u will pass with probablity:", "pass" if(results[0]==1) else "fail")
#print("if u study for 3.9 hours u will pass with probablity:", "pass" if(results[1]==1) else "fail")
#print("if u study for 1 hour u will pass with probablity:", "pass" if(results[2]==1) else "fail")
#print("if u study for 6 hours u will pass with probablity:", "pass" if(results[3]==1) else "fail")