import csv
import streamlit as st
import pickle
import plotly
from plotly import graph_objects as go

def get_clean_data():

    data = []
    #open and read the csv file
    with open(r'C:\Users\NGCS\Documents\Breast Cancer Prediction\data\data.csv',mode='r') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            if 'Unamed: 32' in row:
                del row['Unamed: 32']
            if 'id' in row:
                del row['id']
            row['diagnosis'] = 1 if row['diagnosis'] == 'M' else 0
            data.append(row)
    return data


def find_max_value(data,key):
    max_value = max(float(row[key]) for row in data if key in row)
    return max_value

def find_mean_value(data,key):
    values = [float(row[key]) for row in data if key in row]
    mean_value = sum(values)/len(values) if values else 0
    return mean_value

def add_sidebar():
    st.sidebar.header('Cell Nuclei Measurement')

    data = get_clean_data()

    slide_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label,key in slide_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0.0,
            max_value = find_max_value(data,key),
            value = find_mean_value(data,key),
        )
    return input_dict


def to_numeric(value):
    try:
        return float(value)
    except (ValueError,TypeError):
        return None

def get_scaled_values(input_dict):
    data = get_clean_data()

    feature_columns = [key for key in data[0].keys() if key != 'diagnosis']

    min_values = {key:float('inf') for key in feature_columns}
    max_values = {key:float('-inf') for key in feature_columns}

    for row in data:
        for key in feature_columns:
            value = to_numeric(row.get(key))
            if value is not None:
                min_values[key] = min(value,min_values[key])
                max_values[key] = max(value,max_values[key])
    
    scaled_dict = {}
    for key,value in input_dict.items():
        value = to_numeric(value)
        if value is not None and key in min_values and key in max_values:
            min_val = min_values[key]
            max_val = max_values[key]

            scaled_value = (value - min_val)/(max_val - min_val) if max_val != min_val else 0
            scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)

    categories = ['Radius', 'Texture', 'Perimeter', 'Area', 
                    'Smoothness', 'Compactness', 
                    'Concavity', 'Concave Points',
                    'Symmetry', 'Fractal Dimension']
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open(r'C:\Users\NGCS\Documents\Breast Cancer Prediction\model\model.pkl','rb'))
    scaler = pickle.load(open(r'C:\Users\NGCS\Documents\Breast Cancer Prediction\model\scaler.pkl','rb'))

    input_list = [float(value) for value in input_data.values()]
    input_array = [input_list]

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)
    prediction_proba = model.predict_proba(input_array_scaled)

    st.subheader('Cell Cluster Prediction')
    st.write('The Cell Cluster is:')

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis Malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("probability of being benign: ", prediction_proba[0][0])
    st.write("probability of being malicious: ", prediction_proba[0][1])
    
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


def main():
    st.set_page_config(
        page_title='Breast Cancer Predictor',
        page_icon='female-doctor:',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    input_data = add_sidebar()

    with st.container():
        st.title('Breast Cancer Predictor')
        st.write("Please connect this app to your cytology lab to help diagnose breast cancer form your tissue sample. This app predicts using a machine learning model whether a breast mass is benign or malignant based on the measurements it receives from your cytosis lab. You can also update the measurements by hand using the sliders in the sidebar. ")

    col1,col2 = st.columns([4,1])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

if __name__ == '__main__':
    main()