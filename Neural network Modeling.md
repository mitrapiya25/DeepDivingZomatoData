

```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
```


```python
restaurant_data= pd.read_csv("Resources/zomato.csv",encoding="ISO-8859-1")
X = restaurant_data[["Country Code","Votes"]]
y = restaurant_data["Cuisines"]
```


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
```


```python
from sklearn.preprocessing import StandardScaler

# Create a StandardScater model and fit it to the training data
X_scaler = StandardScaler().fit(X_train)
```


```python
# Transform the training and testing data using the X_scaler

X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
```


```python
from keras.utils import to_categorical

# One-hot encoding
label_encoder = LabelEncoder()
label_encoder.fit(y)
encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)
y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test)
```


```python
# first, create a normal neural network with 2 inputs, 6 hidden nodes, and 2 outputs
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(units=6, activation='relu', input_dim=2))
model.add(Dense(units=49, activation='softmax'))
```


```python
# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```


```python
# Fit the model to the training data
model.fit(
    X_train_scaled,
    y_train_categorical,
    epochs=100,
    shuffle=True,
    verbose=2
)
```

    Epoch 1/100
     - 1s - loss: 3.5515 - acc: 0.1314
    Epoch 2/100
     - 0s - loss: 2.3096 - acc: 0.3559
    Epoch 3/100
     - 0s - loss: 1.6988 - acc: 0.5287
    Epoch 4/100
     - 0s - loss: 1.6136 - acc: 0.5305
    Epoch 5/100
     - 0s - loss: 1.5801 - acc: 0.5305
    Epoch 6/100
     - 0s - loss: 1.5605 - acc: 0.5305
    Epoch 7/100
     - 0s - loss: 1.5477 - acc: 0.5547
    Epoch 8/100
     - 0s - loss: 1.5394 - acc: 0.5562
    Epoch 9/100
     - 0s - loss: 1.5336 - acc: 0.5563
    Epoch 10/100
     - 0s - loss: 1.5300 - acc: 0.5563
    Epoch 11/100
     - 0s - loss: 1.5271 - acc: 0.5556
    Epoch 12/100
     - 0s - loss: 1.5247 - acc: 0.5566
    Epoch 13/100
     - 0s - loss: 1.5224 - acc: 0.5567
    Epoch 14/100
     - 0s - loss: 1.5211 - acc: 0.5563
    Epoch 15/100
     - 0s - loss: 1.5198 - acc: 0.5563
    Epoch 16/100
     - 0s - loss: 1.5179 - acc: 0.5563
    Epoch 17/100
     - 0s - loss: 1.5168 - acc: 0.5561
    Epoch 18/100
     - 0s - loss: 1.5158 - acc: 0.5562
    Epoch 19/100
     - 0s - loss: 1.5149 - acc: 0.5566
    Epoch 20/100
     - 1s - loss: 1.5135 - acc: 0.5563
    Epoch 21/100
     - 0s - loss: 1.5125 - acc: 0.5565
    Epoch 22/100
     - 0s - loss: 1.5118 - acc: 0.5565
    Epoch 23/100
     - 0s - loss: 1.5108 - acc: 0.5563
    Epoch 24/100
     - 0s - loss: 1.5096 - acc: 0.5559
    Epoch 25/100
     - 0s - loss: 1.5092 - acc: 0.5562
    Epoch 26/100
     - 0s - loss: 1.5079 - acc: 0.5565
    Epoch 27/100
     - 0s - loss: 1.5072 - acc: 0.5562
    Epoch 28/100
     - 0s - loss: 1.5065 - acc: 0.5566
    Epoch 29/100
     - 0s - loss: 1.5056 - acc: 0.5566
    Epoch 30/100
     - 0s - loss: 1.5047 - acc: 0.5559
    Epoch 31/100
     - 0s - loss: 1.5042 - acc: 0.5566
    Epoch 32/100
     - 0s - loss: 1.5034 - acc: 0.5566
    Epoch 33/100
     - 0s - loss: 1.5028 - acc: 0.5561
    Epoch 34/100
     - 0s - loss: 1.5021 - acc: 0.5566
    Epoch 35/100
     - 0s - loss: 1.5013 - acc: 0.5566
    Epoch 36/100
     - 0s - loss: 1.5004 - acc: 0.5566
    Epoch 37/100
     - 0s - loss: 1.4999 - acc: 0.5566
    Epoch 38/100
     - 0s - loss: 1.4992 - acc: 0.5565
    Epoch 39/100
     - 0s - loss: 1.4986 - acc: 0.5566
    Epoch 40/100
     - 0s - loss: 1.4980 - acc: 0.5566
    Epoch 41/100
     - 0s - loss: 1.4973 - acc: 0.5566
    Epoch 42/100
     - 0s - loss: 1.4969 - acc: 0.5566
    Epoch 43/100
     - 1s - loss: 1.4966 - acc: 0.5566
    Epoch 44/100
     - 0s - loss: 1.4961 - acc: 0.5566
    Epoch 45/100
     - 0s - loss: 1.4954 - acc: 0.5566
    Epoch 46/100
     - 0s - loss: 1.4952 - acc: 0.5566
    Epoch 47/100
     - 0s - loss: 1.4948 - acc: 0.5566
    Epoch 48/100
     - 0s - loss: 1.4942 - acc: 0.5588
    Epoch 49/100
     - 0s - loss: 1.4939 - acc: 0.5590
    Epoch 50/100
     - 1s - loss: 1.4938 - acc: 0.5590
    Epoch 51/100
     - 1s - loss: 1.4935 - acc: 0.5593
    Epoch 52/100
     - 0s - loss: 1.4931 - acc: 0.5595
    Epoch 53/100
     - 0s - loss: 1.4927 - acc: 0.5595
    Epoch 54/100
     - 0s - loss: 1.4924 - acc: 0.5595
    Epoch 55/100
     - 0s - loss: 1.4923 - acc: 0.5597
    Epoch 56/100
     - 0s - loss: 1.4920 - acc: 0.5597
    Epoch 57/100
     - 0s - loss: 1.4922 - acc: 0.5595
    Epoch 58/100
     - 0s - loss: 1.4915 - acc: 0.5597
    Epoch 59/100
     - 0s - loss: 1.4915 - acc: 0.5597
    Epoch 60/100
     - 0s - loss: 1.4913 - acc: 0.5597
    Epoch 61/100
     - 0s - loss: 1.4913 - acc: 0.5597
    Epoch 62/100
     - 0s - loss: 1.4911 - acc: 0.5597
    Epoch 63/100
     - 0s - loss: 1.4909 - acc: 0.5597
    Epoch 64/100
     - 1s - loss: 1.4908 - acc: 0.5597
    Epoch 65/100
     - 0s - loss: 1.4907 - acc: 0.5597
    Epoch 66/100
     - 1s - loss: 1.4904 - acc: 0.5597
    Epoch 67/100
     - 0s - loss: 1.4904 - acc: 0.5597
    Epoch 68/100
     - 0s - loss: 1.4905 - acc: 0.5597
    Epoch 69/100
     - 0s - loss: 1.4902 - acc: 0.5597
    Epoch 70/100
     - 0s - loss: 1.4900 - acc: 0.5597
    Epoch 71/100
     - 0s - loss: 1.4902 - acc: 0.5597
    Epoch 72/100
     - 0s - loss: 1.4901 - acc: 0.5597
    Epoch 73/100
     - 0s - loss: 1.4900 - acc: 0.5597
    Epoch 74/100
     - 0s - loss: 1.4897 - acc: 0.5597
    Epoch 75/100
     - 0s - loss: 1.4900 - acc: 0.5597
    Epoch 76/100
     - 0s - loss: 1.4897 - acc: 0.5597
    Epoch 77/100
     - 0s - loss: 1.4898 - acc: 0.5597
    Epoch 78/100
     - 1s - loss: 1.4894 - acc: 0.5597
    Epoch 79/100
     - 0s - loss: 1.4896 - acc: 0.5597
    Epoch 80/100
     - 0s - loss: 1.4893 - acc: 0.5597
    Epoch 81/100
     - 0s - loss: 1.4892 - acc: 0.5597
    Epoch 82/100
     - 0s - loss: 1.4893 - acc: 0.5597
    Epoch 83/100
     - 0s - loss: 1.4890 - acc: 0.5597
    Epoch 84/100
     - 0s - loss: 1.4886 - acc: 0.5597
    Epoch 85/100
     - 0s - loss: 1.4888 - acc: 0.5597
    Epoch 86/100
     - 0s - loss: 1.4889 - acc: 0.5597
    Epoch 87/100
     - 0s - loss: 1.4890 - acc: 0.5597
    Epoch 88/100
     - 0s - loss: 1.4887 - acc: 0.5597
    Epoch 89/100
     - 1s - loss: 1.4887 - acc: 0.5597
    Epoch 90/100
     - 1s - loss: 1.4884 - acc: 0.5597
    Epoch 91/100
     - 0s - loss: 1.4884 - acc: 0.5597
    Epoch 92/100
     - 0s - loss: 1.4885 - acc: 0.5597
    Epoch 93/100
     - 0s - loss: 1.4884 - acc: 0.5597
    Epoch 94/100
     - 0s - loss: 1.4885 - acc: 0.5595
    Epoch 95/100
     - 0s - loss: 1.4880 - acc: 0.5597
    Epoch 96/100
     - 0s - loss: 1.4882 - acc: 0.5597
    Epoch 97/100
     - 0s - loss: 1.4880 - acc: 0.5597
    Epoch 98/100
     - 0s - loss: 1.4881 - acc: 0.5597
    Epoch 99/100
     - 0s - loss: 1.4878 - acc: 0.5597
    Epoch 100/100
     - 0s - loss: 1.4880 - acc: 0.5597
    




    <keras.callbacks.History at 0x1dd6e2ffac8>




```python
model_loss, model_accuracy = model.evaluate(
    X_test_scaled, y_test_categorical, verbose=2)
print(
    f"Normal Neural Network - Loss: {model_loss}, Accuracy: {model_accuracy}")
```

    Normal Neural Network - Loss: 1.5359347563492793, Accuracy: 0.5598827472683573
    
