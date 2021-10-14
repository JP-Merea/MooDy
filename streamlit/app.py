import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="MooDy",
    page_icon="logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)
#######################################
# css_path
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

local_css("streamlit/style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')
remote_css('https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css')

#######################################
# containers
header_container = st.container()	
results_container = st.container()	
dolar_widget_container = st.container()	
#######################################
# variables
alza = 100
estable = 3
baja = -5

#######################################
# body
with header_container:

    # for example a logo or a image that looks like a website header
    st.image('logo.png')

    # different levels of text you can include in your app
    st.title("Covid, Health and Economy Dilemma")
    st.header("Correlation between social mood and local economy variable(Dolar)")
    st.subheader("Le Wagon")
 
with results_container:
    col_1, col_2 = st.columns(2)
    
    filename = st.text_input('Enter a file path:')
    try:
        with open(filename) as input:
            st.text(input.read())
    except FileNotFoundError:
        st.error('File not found.')
    RESULTS = (f"""
    <head><link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.1/dist/css/bootstrap.min.css" integrity="sha384-F3w7mX95PdgyTmZZMECAngseQB83DfGTowi0iMjiWaeVhAn4FJkqJByhZMI3AhiU" crossorigin="anonymous"></head>
    <body class='results-html'>
        <div class='container'>
            <row>
                <div class="col-12">
                    <div class="row"><span class='results alza'>Alza: %{alza}</span></div>
                    <br>
                    <div class="row"><span class='results estable'>Estable: %{estable}</span></div>
                    <br>
                    <div class="row"><span class='results baja'>Baja: %{baja}</span></div>
                </div>
            </row>
        </div>
    </body>
    """)

    col_1.write(RESULTS, unsafe_allow_html=True)

    @st.cache
    def get_line_chart_data():

        return pd.DataFrame(
                np.random.randn(20, 3),
                columns=['a', 'b', 'c']
            )

    df = get_line_chart_data()

    col_2.line_chart(df)
    

with dolar_widget_container:
    
    components.iframe("https://dolar-plus.com/api/widget")
    
#url = 'https://taxifare.lewagon.ai/predict'

#params = dict(
#    pickup_datetime=pickup_datetime,
#    pickup_longitude=pickup_longitude,
#    pickup_latitude=pickup_latitude,
#    dropoff_longitude=dropoff_longitude,
#    dropoff_latitude=dropoff_latitude,
#    passenger_count=passenger_count)

#response = requests.get(url, params=params)

#prediction = response.json()

#pred = prediction['prediction'] 
#pred