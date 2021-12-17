import pandas as pd
import streamlit as st
import numpy as np
import model


final_models,model_cvs,names = model.start()

question = st.selectbox("What question do you want?", ["How tall is this?", "how wide is this?"], index=0)
answer = st.text_area("What's the answer" , value="")

if st.button("Is this answer helpful?", key=None, help=None, on_click=None, args=None, kwargs=None):
  df,prediction = model.predict(question,answer,final_models,model_cvs,names)
  st.table(df)
  st.text(f'Your answer was {prediction}')