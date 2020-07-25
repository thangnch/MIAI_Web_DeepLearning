from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
import numpy as np
import glob
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

print("Build model....")

# Su dung CGG16
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

# Dong bang cac layer
for layer in model_vgg16_conv.layers:
    layer.trainable = False

# Tao model
input = Input(shape=(200,200,3),name = 'image_input')
output_vgg16_conv = model_vgg16_conv(input)

# Them cac layer FC va Dropout
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name='fc2')(x)
x = Dropout(0.5)(x)
x = Dense(2, activation='softmax', name='predictions')(x)

#Compile
my_model = Model(input=input, output=x)
my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Load du lieu
print("Load Data....")
# Load data
data_folder = 'data/'
folder_list = ['0','1']

data = []
label = []

for folder in folder_list:
    for file in glob.glob(data_folder + folder + "/*"):
        # Read
        image = cv2.imread(file)
        # Resize
        image = cv2.resize(image,dsize=(200,200))
        # Add to data
        data.append(image)
        label.append(folder)

print("Data length=",len(data))

# Split train_test
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(label)
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)

# Change to numpy array
label = onehot_encoded
data = np.array(data)

# Fit to model
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)
hist = my_model.fit(X_train, y_train, epochs=2, validation_data=(X_test, y_test), verbose=1)


# Save model
my_model.save("cool_model.h5")
print("Finish model!")

