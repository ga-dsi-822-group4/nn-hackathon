import numpy as np
import pandas as pd
from operator import mod
import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from keras.utils import load_img, img_to_array

# @st.cache
def load_model():
  loaded_model = tf.keras.models.load_model('./models/cnn_1_exported')
  return loaded_model

model = load_model()

st.title('Hotdog or NotHotdog?')

st.subheader('Is your image a hotdog or not?')

uploaded_file = st.file_uploader(label='Upload your image here')
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=(300, 300), color_mode='grayscale')
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = image.astype('float32')
    image /= 255.0




if st.button('Submit'):
    st.image(uploaded_file, width=300)
    pred = model.predict(image)
    st.write('Your image prediction is: ', 'hotdog' if pred[0] > 0.5 else 'nothotdog')
