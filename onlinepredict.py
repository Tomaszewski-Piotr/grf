import streamlit as st
import base64
from io import BytesIO
import pandas as pd
from joblib import load
import common
import re

porosity_options = ('low', 'high')

def get_classification(file_prefix, data):
    pred_clf = load(common.out_file(file_prefix + '_classifier.joblib'))
    pred_enc = load(common.out_file(file_prefix + '_encoder.joblib'))
    prediction = pred_clf.predict(data)
    result = pd.DataFrame(pred_enc.inverse_transform(prediction))
    top_res = result[0].value_counts().idxmax()
    return top_res

def get_regression(file_prefix, data):
    reg_clf = load(common.out_file(file_prefix + '.joblib'))
    prediction = reg_clf.predict(data)
    titles = []
    for i in range (0, prediction.shape[1]):
        titles.append(file_prefix+'_'+str(i+1))
    return pd.DataFrame(prediction, columns=titles)

def fetch_relevant_classes(porosity):
    # using re + search()
    # to get string with substring
    classes = load(common.out_file('classes.joblib'))
    res = [x for x in classes if re.search(porosity, x)]
    return res

def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Results') #, float_format="%.2f")
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_evaluation():
    with open(common.out_file('output.xlsx'), 'rb') as fh:
        buf = BytesIO(fh.read())
    b64 = base64.b64encode(buf.getvalue())
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="evaluation.xlsx"> click this link</a>'

st.markdown('To see the results of the evaluation of the underlying ML models: ' + get_evaluation(), unsafe_allow_html=True)

file = st.file_uploader(label="Upload file for prediction", accept_multiple_files=False, type=["csv"], help="Select file to perform prediction on")

#if selection made
if file is not None:
    #prepare for possible multiple files

    input_df = pd.read_csv(file, header=None) # we asure header and one line

    porosity = get_classification('porosity_pred', input_df)

    por_option = st.radio(
               'Porosity',
                porosity_options,
                help='Select porosity',
                index=porosity_options.index(porosity))

    # st.write('You selected:', option)
    classes = fetch_relevant_classes(por_option)
    predicted_class = get_classification(por_option+'_porosity', input_df)

    class_option = st.radio(
                   'Class',
                    classes,
                    help='Select class',
                    index=classes.index(predicted_class))

    reg = get_regression(class_option, input_df)
    st.dataframe(reg)

    st.download_button(
              label="Download result as an Excel file",
              data=to_excel(reg),
              file_name='result.xlsx',
              mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
              )

