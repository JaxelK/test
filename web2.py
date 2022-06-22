import streamlit as st
import cv2

st.title("68 LandMarks place")
Run=st.checkbox('run')
FRAME_WINDOW = st.image([])
cam=cv2.VideoCapture(1)

while Run:
    ret,frame=cam.read()
    FRAME_WINDOW.image(frame)
else:
    st.write("stopped")
