import pandas as pd
import matplotlib.pyplot as plt

data = {
    'Hours':[1,2,3,4,5],
    'Marks':[15,25,35,45,55]
}

df = pd.DataFrame(data)

print(df)

plt.scatter(df['Hours'], df['Marks'])

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")

plt.show()



import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    'Hours':[1,2,3,4,5,6,7,8,9,10],
    'Marks':[15,25,35,45,55,65,75,85,95,100]
}

df = pd.DataFrame(data)

X = df[['Hours']]
y = df['Marks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = r2_score(y_test, predictions)

print("Accuracy:", accuracy)

new_student = [[8]]

predicted_marks = model.predict(new_student)

print("Predicted Marks for 8 Hours Study:", predicted_marks[0])

plt.scatter(X, y)

plt.plot(X, model.predict(X))

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")

plt.show()