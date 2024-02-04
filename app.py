import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import time
import platform
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# TITLE
st.title("I KNOW EVERYONE CAN FIND IF THE HORSE PIC WILL BE SHOWN !\n IT IS JUST MODEL FOR FUN :> ")

# OIC UOLOADs
file = st.file_uploader("PIC UPLOAD", type=['jpg', 'png', 'jpeg', 'avif', 'svg'])


# PROCESSIMG....
if file:
    st.image(file)

    # IMG CONVERT .....
    img = PILImage.create(file)

    # MODEL...
    model = load_learner('horse_model.pkl')


    # PREDICTION
    pred, pred_id, prob = model.predict(img)
    with st.spinner(text="In progress..... "):
        time.sleep(3)
        st.success(f"PREDICTION : {pred}")
        st.info(f"PROBABILITY : {prob[pred_id]*100:.1f}%")



    # # PLOT
    # fig = px.bar(x=prob*100, y=model.dls.vocab)
    # st.plotly_bar(fig)

























