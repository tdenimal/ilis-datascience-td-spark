#import packages
import streamlit as st
import numpy as np
import matplotlib.image as mpimg
import os
import PIL.Image
import requests
from io import BytesIO
from inference import img_inference




## Frontend Design

#STYLES
with open("style.css") as f:
  st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

#SIDEBAR
st.sidebar.title("Spark Xray")

st.sidebar.markdown("""
  An inference API for pre-screening upper-respitory infectious diseases based on Chest XRays (CXR) images.
  """, unsafe_allow_html=True,)


st.sidebar.info(
        " [View source code on GitHub](https://github.com/tdenimal/ilis-datascience-td-spark)."
    )
st.sidebar.header("About")

st.sidebar.markdown("""
  This model builds on dataset provided by [NIHCC](https://nihcc.app.box.com/v/ChestXray-NIHCC). 
  <br><br>
  **Maintained by:** <br>Thomas DENIMAL<br>
  [tdenimal.github.io](https://tdenimal.github.io)
  """, unsafe_allow_html=True,)



#MAIN CONTENT
#variable paths
images_path = '../data/output/'

#Selectbox for test images
test_images = os.listdir(images_path)
test_image = st.selectbox('Please select a test image:', test_images)
file_path = images_path + test_image

#Img inference
img_inference(file_path)

