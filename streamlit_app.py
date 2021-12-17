import pandas as pd
import streamlit as st
import numpy as np
import model
from PIL import Image


final_models,model_cvs,names = model.start()

st.title(f'TEAM BRAZIL ANSWER CHECK')


question = st.selectbox("Question", ["Does this come with batteries?", "How wide is this?"], index=0)
answer = st.text_area("Answer" , value="")

if st.button("Is this answer helpful?", key=None, help=None, on_click=None, args=None, kwargs=None):
    df,prediction = model.predict(question,answer,final_models,model_cvs,names)
    
    st.markdown("""---""")
    st.header(f'Your answer was {prediction}')
    st.markdown("""---""")


    st.subheader("Summary")
    st.table(df)
    st.text(f'Veto power.')
    st.text(f"We just need one model to flag your answer as unhelpful.")
    st.text(f"1 - Unhelpful")
    st.text(f"0 - Helpful!")
    st.markdown("""---""")


image = Image.open('Brazil.png')
st.image(image, caption='Team Brazil')