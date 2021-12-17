import pandas as pd
import streamlit as st
import numpy as np
import model


final_models,model_cvs,names = model.start()

question = st.selectbox("What question do you want?", ["How tall is this?", "how wide is this?"], index=0)
answer = st.text_area("What's the answer" , value="")

if st.button("Is this answer helpful?", key=None, help=None, on_click=None, args=None, kwargs=None):
    df,prediction = model.predict(question,answer,final_models,model_cvs,names)
    
    st.header(f'Your answer was {prediction}')
    st.table(df)
    st.text(f'Veto power.')
    st.text(f"Your answer will be unhelpful if any of the models decide it is unhelfpul.")
    st.text(f"1 - Unhelpful")
    st.text(f"0 - Helpful!")
