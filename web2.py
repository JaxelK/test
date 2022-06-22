import streamlit as st
import cv2

image_placeholder = st.empty()
if st.button('Start'):
    FRAME_WINDOW = st.image([])
    cam=cv2.VideoCapture(1)
    while run:
        ret,frame=cam.read()
        FRAME_WINDOW.image(frame)
    else:
        srt.write('stopped')
