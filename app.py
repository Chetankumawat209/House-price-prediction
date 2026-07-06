import pickle
import pandas as pd
import streamlit as st

# st.set_page_config(page_title="India House Price Predictor", layout="centered")

# -------------------------------------------------------
@st.cache_resource
def load_bundle():
    with open("House_price_model.pkl", "rb") as f:
        return pickle.load(f)

bundle = load_bundle()
model = bundle["model"]
ohe = bundle["ohe"]
mlb = bundle["mlb"]
oe = bundle["oe"]
scaler_X = bundle["scaler_X"]
scaler_y = bundle["scaler_y"]
cat_cols = bundle["cat_cols"]
num_cols = bundle["num_cols"]
feature_columns = bundle["feature_columns"]
amenity_classes = [a for a in bundle["amenity_classes"] if a != "None"]

st.title("India House Price Predictor")
st.write("Fill in the property details below to estimate the price (in Lakhs ₹).")


# ---------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox("City", bundle["city_options"])
        property_type = st.selectbox("Property Type", bundle["property_type_options"])
        bhk = st.number_input("BHK", min_value=1, max_value=5, value=2, step=1)
        size_sqft = st.number_input("Size (in SqFt)", min_value=300, max_value=4700, value=1000, step=50)
        floor_no = st.number_input("Floor Number", min_value=0, max_value=29, value=2, step=1)
        age = st.number_input("Age of Property (years)", min_value=1, max_value=45, value=5, step=1)

    with col2:
        furnished_status = st.selectbox("Furnished Status", ["Unfurnished", "Semi-furnished", "Furnished"])
        transport = st.selectbox("Public Transport Accessibility", ["Low", "Medium", "High"])
        facing = st.selectbox("Facing", bundle["facing_options"])
        parking = st.selectbox("Parking Space", ["No", "Yes"])
        security = st.selectbox("Security", ["No", "Yes"])
        nearby_schools = st.number_input("Nearby Schools", min_value=0, max_value=10, value=3, step=1)
        nearby_hospitals = st.number_input("Nearby Hospitals", min_value=0, max_value=10, value=3, step=1)

    amenities = st.multiselect("Amenities", amenity_classes)

    submitted = st.form_submit_button("Predict Price")

#Pred
# ---------------------------------------------------------
if submitted:
    raw = pd.DataFrame([{
        "City": city,
        "Property_Type": property_type,
        "BHK": bhk,
        "Size_in_SqFt": size_sqft,
        "Furnished_Status": furnished_status,
        "Floor_No": floor_no,
        "Age_of_Property": age,
        "Nearby_Schools": nearby_schools,
        "Nearby_Hospitals": nearby_hospitals,
        "Public_Transport_Accessibility": transport,
        "Parking_Space": parking,
        "Security": security,
        "Facing": facing,
        "Amenities": amenities if amenities else ["None"],
    }])

    # One-hot encode categorical columns
    ohe_out = ohe.transform(raw[cat_cols])
    raw[ohe.get_feature_names_out()] = ohe_out
    raw = raw.drop(columns=cat_cols)

    # Multi-label binarize amenities
    mlb_out = pd.DataFrame(mlb.transform(raw["Amenities"]), columns=mlb.classes_)
    raw[mlb.classes_] = mlb_out
    raw = raw.drop(columns="Amenities")

    # Ordinal encode furnished status / transport accessibility
    raw[["Furnished_Status", "Public_Transport_Accessibility"]] = oe.transform(
        raw[["Furnished_Status", "Public_Transport_Accessibility"]]
    )

    # Scale numeric columns
    raw[num_cols] = scaler_X.transform(raw[num_cols])

    # Align column order exactly as during training
    raw = raw[feature_columns]

    # Predict (model output is scaled 0-1) then inverse-transform to Lakhs
    pred_scaled = model.predict(raw)
    pred_lakhs = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]

    st.success(f"### Estimated Price: ₹ {pred_lakhs:,.2f} Lakhs")
