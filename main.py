#Support vector machines (SVMs) are supervised machine learning algorithms for outlier detection, regression, and classification that are both powerful and adaptable. Sklearn SVMs are commonly employed in classification tasks because they are particularly efficient in high-dimensional fields.
from sklearn import svm

#holds data for video game ratings
student_data = [ #must be 4x4, 5x5, or 6x6, etc.
    [8, 1, 15, 7], #1
    [8, 0, 14, 8], #2
    [9, 1, 15, 8], #3
    [7, 1, 15, 8], #4
]

#This list game_data corresponds to the labels or output values associated with each set of student features. Each label represents the game recommendation for the respective student in student_data, must maintain array length with matrix length
game_data = [1, 2, 3, 4] 

#creates an instance of the Support Vector Classification (SVC) model. SVC is a type of Support Vector Machine used for classification tasks.
recommendation_model = svm.SVC()


#fit()
#– It calculates the parameters or weights on the training data (e.g. parameters returned by coef() in case of Linear Regression) and saves them as an internal object state.
recommendation_model.fit(student_data, game_data)

#predict()
#– Use the above-calculated weights on the test data to make the predictions.
print(recommendation_model.predict([ [9, 0, 15, 7] ])) 
# prints [3] 

#must maintain array length with matrix length



videogame_data = [
  [8,3,6,1], 
  [6,2,7,0], 
  [5,1,9,1], 
  [7,8,6,0], 
]

game = [1,2,3,4]

model = svm.SVC()
model.fit(videogame_data, game)

print(model.predict([ [8,3,9,8,6] ]))