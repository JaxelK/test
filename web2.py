import streamlit as st
import cv2

image_placeholder = st.empty()
if st.button('Start'):
    video = cv2.VideoCapture('https://videos3.earthcam.com/fecnetwork/9974.flv/chunklist_w372707020.m3u8?__fvd__')
    while True:
        success, image = video.read()
        image_placeholder.image(image)