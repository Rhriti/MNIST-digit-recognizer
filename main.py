import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# mnsit=tf.keras.datasets.mnist
# #we split knwn dta as trainign and testiung daa
# (x_train,y_train),(x_test,y_test)=mnsit.load_data()


# print(x_train,x_test)

# x_train=tf.keras.utils.normalize(x_train,axis=1)
# x_test=tf.keras.utils.normalize(x_test,axis=1)


# model=tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# #rectify linear unit
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dense(10,activation='softmax'))#all neurons add up to one


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train,epochs=7)
# model.save('handwritten.model')

model=tf.keras.models.load_model('handwritten.model')
# loss,accuracy=model.evaluate(x_test,y_test)

# print(loss)
# print(accuracy)

image_number=1
while os.path.isfile(f'{image_number}.png'):
    try:
        img=cv2.imread(f"{image_number}.png")[:,:,0]
        img=np.invert(np.array([img]))
        prediction=model.predict(img)
        print(f"this digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print('error')
    finally:
        image_number+=1
        print('yes')