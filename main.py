import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Dropout,MaxPooling2D,Conv2D 
import numpy as np 
import matplotlib.pyplot as plt 


(x_train,y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

# Data PreProcessing
x_train=x_train/255.0 
x_test=x_test/255.0 

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)


#Build CNN Model

model=Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128,activation='relu'),
    Dropout(0.5),
    Dense(10,activation='softmax')
])

#Compile Model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Train Model

history=model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test))

# Evaluate Model
test_loss,test_acc=model.evaluate(x_test,y_test)
print('Test Accuracy:',test_acc)

# Plot Accuracy
plt.plot(history.history['accuracy'],  label='Train Accuracy')
plt.plot(history.history['val_accuracy'] , label='Validation Accuracy')
plt.legend()
plt.show()

#Prediction
index=0
prediction=model.predict(x_test[index].reshape(1,28,28,1))
predicted_digit=np.argmax(prediction)

plt.imshow(x_test[index].reshape(1,28,28,1),cmap='gray')
plt.title(f'Predicted Digit:{predicted_digit}')
plt.axis('off')
plt.show()

model.save('digit_recoginition')