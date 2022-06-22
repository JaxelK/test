import streamlit as st
import cv2

st.title("68 LandMarks place")
Run=st.checkbox('run')
FRAME_WINDOW = st.image([])
cam=cv2.VideoCapture(0)

while Run:
    ret,frame=cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)
else:
    st.write("stopped")
