
import tensorflow as tf 
from keras.utils import image_dataset_from_directory,load_img,img_to_array
from keras.models import Sequential,load_model
from keras.layers import Dense,MaxPooling2D,Flatten,Conv2D,Dropout,BatchNormalization,Input,GlobalAveragePooling2D
import numpy as np 
from tensorflow.keras.applications import MobileNetV2


train_data = image_dataset_from_directory(r"E:\dataset\chcek\train\train",image_size=(224, 224))

test_data = image_dataset_from_directory(r"E:\dataset\chcek\test\test",image_size=(224, 224))

class_names = train_data.class_names
print(class_names)


print(train_data.class_names)

train_data = train_data.map(lambda x, y: (x/255.0, y))
test_data = test_data.map(lambda x, y: (x/255.0, y))



base_model = MobileNetV2(input_shape=(224,224,3),include_top=False)
base_model.trainable = False

model = Sequential()

model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(class_names), activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data, validation_data=test_data, epochs=10)

img_path1 = input("Enter image path: ").strip().strip('""')

img1 = load_img(img_path1, target_size=(224,224))
img_array1 = img_to_array(img1)

img_array1 = img_array1 / 255.0
img_array1 = np.expand_dims(img_array1, axis=0)
prediction1= model.predict(img_array1)

predicted_class1 = np.argmax(prediction1,axis=1)[0]
animal_name1 = class_names[predicted_class1]

print("Predicted Animal:", animal_name1)










