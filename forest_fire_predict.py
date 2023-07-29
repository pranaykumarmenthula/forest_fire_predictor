import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model.pkl','rb'))


def predict_forest(oxygen,humidity,temperature):
    input=np.array([[oxygen,humidity,temperature]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    
    st.title("Forest Fire Prediction")
    oxygen = st.text_input("Oxygen")
    humidity = st.text_input("Humidity")
    temperature = st.text_input("Temperature")

    if st.button("Predict"):
        output=predict_forest(oxygen,humidity,temperature)
        st.success('The probability of fire taking place is {}'.format(output))

if __name__=='__main__':
    main()
