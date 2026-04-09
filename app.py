import streamlit as st
import pickle

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #00FFAA;'>🚗 Car Price Predictor</h1>
    <p style='text-align: center;'>Predict used car prices using Machine Learning</p>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = pickle.load(open('./saved_models/RandomForestRegressor.pkl', 'rb'))
scaler = pickle.load(open('./saved_scaling/scaler.pkl', 'rb'))

# -------------------- FORMAT FUNCTION --------------------
def format_value(value):
    if value >= 10000000:
        return f"{value/10000000:.2f} Cr"
    elif value >= 100000:
        return f"{value/100000:.2f} Lakhs"
    else:
        return str(value)

# -------------------- INPUT SECTION --------------------
st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    vehicle_age = st.number_input("Vehicle Age", min_value=0, value=5)
    km_driven = st.number_input("KM Driven", min_value=0, value=50000)
    mileage = st.number_input("Mileage", value=18.5)

with col2:
    engine = st.number_input("Engine (CC)", value=1197)
    max_power = st.number_input("Max Power", value=82.0)
    seats = st.number_input("Seats", min_value=1, max_value=7, value=5)

# -------------------- CATEGORICAL INPUTS --------------------
seller = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
fuel = st.selectbox("Fuel Type", ["CNG", "Diesel", "Electric", "LPG", "Petrol"])
transmission = st.radio("Transmission", ["Automatic", "Manual"])

# -------------------- PREDICTION --------------------
if st.button("🚀 Predict Car Price"):

    # Validation
    if vehicle_age == 0 or km_driven == 0:
        st.error("Please fill all required fields properly")
    else:
        input_values = []

        # Numeric Features
        input_values.extend([
            vehicle_age,
            km_driven,
            mileage,
            engine,
            max_power,
            seats
        ])

        # Seller Encoding
        if seller == "Dealer":
            input_values.extend([1, 0, 0])
        elif seller == "Individual":
            input_values.extend([0, 1, 0])
        else:
            input_values.extend([0, 0, 1])

        # Fuel Encoding
        fuel_dict = {
            "CNG": [1, 0, 0, 0, 0],
            "Diesel": [0, 1, 0, 0, 0],
            "Electric": [0, 0, 1, 0, 0],
            "LPG": [0, 0, 0, 1, 0],
            "Petrol": [0, 0, 0, 0, 1]
        }
        input_values.extend(fuel_dict[fuel])

        # Transmission Encoding
        if transmission == "Automatic":
            input_values.extend([1, 0])
        else:
            input_values.extend([0, 1])

        # Final Check
        if len(input_values) == 16:
            scaled = scaler.transform([input_values])
            prediction = model.predict(scaled)[0]
            formatted = format_value(prediction)

            st.success(f"Predicted Price: ₹ {formatted}")

            # Feature transparency
            with st.expander("🔍 See Model Input Features"):
                st.write(input_values)

        else:
            st.error("Feature mismatch! Expected 16 inputs.")

# -------------------- INFO --------------------
st.info("""
This prediction is generated using a Random Forest Regression model 
trained on historical used car data.
""")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Made with ❤️ by Khushi Ranka")