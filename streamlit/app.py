import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from pysentimiento import EmotionAnalyzer, SentimentAnalyzer
from PIL import Image


st.set_page_config(
    page_title="MooDy",
    page_icon="streamlit/logo.png",
    layout="wide",
    initial_sidebar_state="expanded", 
)
#######################################
# css

st.markdown(
        f"""
<style>
    .main {{background-color: #101010;}}
    .reportview-container .main .block-container{{
    max-width: 80%;
    padding-top: 5rem;
    padding-right: 5rem;
    padding-left: 5rem;
    padding-bottom: 5rem;
    color: red;
}}
.results-html{{
    background-color: #0e1117;
}}
img{{
    max-width:10%;
    margin-bottom:10px;
    width: 50%;
    margin: 0 auto;
}}
iframe{{
    height: 290px;
}}
.css-10trblm {{color: white;}}
.css-1vgnld3 {{color: white;}}
</style>
""",
        unsafe_allow_html=True,
    )

#######################################
# containers
header_container = st.container()	
extra_container = st.container()
emo_container = st.container()
sent_container = st.container()
results_container = st.container()	
dolar_widget_container = st.container()	
#######################################
# variables
#alza = '100%'
#baja = '-50%'

#######################################
# body
with header_container:

    # img logo
    st.image('streamlit/logo.png')

    # Headers and text
    st.title("Covid, Health and Economy Dilemma")
    st.header("Correlation between social mood and local economy variable(Dolar)")
    st.write("Information has always been a fundamental axis for any organization and institution, \
            from the incorporation of artificial intelligence it has been possible to generate dynamic metrics \
            that facilitate the reading of social humor, interests, and opinions. MooDy works in different level, \
            first analyze if the comment is positive, negative, or neutral. For doing this we work in our threshold to make this difference representative. \
            Then we evaluate the historical value of the dollar, and the model will predict if it’s goes up or down.")
    
    st.set_option('deprecation.showfileUploaderEncoding', False)


d = st.date_input("The emotion in Argentina during covid",datetime.date(2020, 4, 24))
st.write('the emotion of:', d, 'was')


#emotional analyzer 
#emo = st.text_input('Emotional Analyzer', '')
#if emo is not None:
#    emotion_analyzer = EmotionAnalyzer(lang="es")
#    emo = emotion_analyzer.predict(emo)
#    st.write('Your emotion is:', emo)



#sentimental analyzer 
#sent = st.text_input('Sentimental Analyzer', '')
#if sent is not None:
#    sentimental_analyzer = SentimentAnalyzer(lang="es")
#    sent = sentimental_analyzer .predict(emo)
#    st.write('Your sentiment is:', sent)


uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
result = [np.ones((30, 2))]

@st.cache
def get_map_data():

    return pd.DataFrame(
            np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
            columns=['lat', 'lon']
        )

df = get_map_data()

st.map(df)

if uploaded_file is not None:
    filepath = 'streamlit/gru_model.h5'
    data = pd.read_csv(uploaded_file)
    model = load_model (filepath, custom_objects=None, compile=True, options=None)
    X = np.array(data.indice).reshape(1,30)
    X_test_pad = pad_sequences(X, value=-1000., dtype=float, padding='post', maxlen=30)
    result = model.predict(X_test_pad)
    st.write(X_test_pad)
    col_1, col_2= st.columns(2)
    alza = round(float(result[0][-1][0]),2)
    baja = -(round(float(result[0][-1][1]),2))
    col_1.metric("Probabilidad de Alza", round(float(result[0][-1][0]),2), alza)
    col_2.metric("Probabilidad de Baja", round(float(result[0][-1][1]),2), baja)


@st.cache
def get_area_chart_data():
    return pd.DataFrame(
                result[0],
                columns=['alza', 'baja']
            )

df = get_area_chart_data()

st.line_chart(df)



    
components.iframe("https://dolar-plus.com/api/widget")

image = Image.open('streamlit/logo_lewagon.png')
st.image(image, caption='aprobe this project', use_column_width=False)