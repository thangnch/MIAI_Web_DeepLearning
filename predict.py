
import numpy as np
from keras.models import load_model
import cv2
import sys

# Dinh nghia class
class_name = ['ok','xxx']

# Load model da train
my_model = load_model("cool_model.h5")

# Doc anh
image_org = cv2.imread(sys.argv[1])

# Resize
image = image_org.copy()
image = cv2.resize(image, dsize=(200, 200))
# Convert to tensor
image = np.expand_dims(image, axis=0)

#Predict
predict = my_model.predict(image)
print("This picture is: ", class_name[np.argmax(predict)])

# Show image
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
fontScale = 1
color = (255, 0, 0)
thickness = 2

cv2.putText(image_org,class_name[np.argmax(predict)] , org, font,
                    fontScale, color, thickness, cv2.LINE_AA)

cv2.imshow("Picture",image_org)
cv2.waitKey()
