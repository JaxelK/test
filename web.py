# Đoạn code này không cần quan tâm có chạy được trên python hay không, vì nó sẽ chạy trực tiếp trên github, do trên python chưa cài streamlit
import numpy as np
import streamlit as st
import tensorflow as tf
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input as mobilenet_v2_preprocess_input
import cv2
  
st.title("Webcam Live Feed")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while run:
    
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    imgsample = cv2.resize(gray, (108,192), interpolation = cv2.INTER_AREA)
    for rect in rects:
        shape = model.predict(imgsample)
	# Convert it to the NumPy Array
        shape_np = np.zeros((68, 2), dtype="int")
        for i in range(0, 68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)
        shape = shape_np

        # Display the landmarks
        for i, (x, y) in enumerate(shape):
	    # Draw the circle to mark the keypoint 
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
		
    # Display the image
    cv2.imshow('Landmark Detection', image)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write('Stopped')






model = tf.keras.models.load_model("Hue.h5") #model m train

### load file
uploaded_file = st.file_uploader("Choose an image file", type=["jpg","jpeg","png"])

map_dict = {0: 'NORMAL',
            1: 'PNEUMONIA'} #vì của t có 2 laoij là bệnh với ko bị bệnh nên t làm cái này, cái này sẽ tùy vô giải thuật của m
    
 
if uploaded_file is not None:
    # Convert the file
    img = image.load_img(uploaded_file,target_size=(64,64)) #xử lí ảnh theo cách m làm
    st.image(uploaded_file, channels="RGB") #hiển thị ảnh
    img = img_to_array(img)
    img = img.reshape(1,64,64,3)
    img = img.astype('float32')
    img = img/255
        
    #Button: nút dự đoán sau khi up ảnh
    Genrate_pred = st.button("Generate Prediction") 
    
    if Genrate_pred:
    
        prediction = model.predict(img).argmax()
        st.write("**Predicted Label for the image is {}**".format(map_dict [prediction])) ##đưa ra dự đoán viêm phổi hay ko