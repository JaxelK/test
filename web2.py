import streamlit as st
import cv2

st.tittle('68 LandMarks place')
Run=st.checkbox('run')
FRAME_WINDOW = st.image([])
cam=cv2.VideoCapture(1)
while run:
    ret,frame=cam.read()
    FRAME_WINDOW.image(frame)
else:
    srt.write('stopped')
