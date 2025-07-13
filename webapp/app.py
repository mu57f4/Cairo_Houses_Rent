import streamlit as st
import pandas as pd
import joblib
import utils.preprocessing as preprocessing

st.title("🏠 Cairo Houses Rent Prediction")

# Load the pre-trained model
@st.cache_resource
def load_model():
    return joblib.load('webapp/utils/rf_model.pkl')

@st.cache_resource
def load_pipeline():
    return joblib.load('webapp/utils/scaler.pkl')

model = load_model()
scaler_pipeline = load_pipeline()

# Input features
house_options = ["Apartment | شقة", "Duplex | دوبلكس", "Penthouse | بنتهاوس", "Room | غرفة", "Studio | استوديو"]
house_type = st.selectbox(
    "House Type | نوع الشقة",
    options=house_options,
    index=0
)
house_type = [int(house_type.__contains__(t)) for t in house_options]

full_address = st.text_input("Address | العنوان", placeholder="46 شارع عباس العقاد، القاهرة")

level = st.number_input("Floor Level | الطابق", min_value=0, max_value=12, value=0)
furnished = st.selectbox("Furnished | مفروش", options=["Yes | نعم", "No | لا"], index=0)
furnished = 1 if furnished == "Yes | نعم" else 0

bedrooms = st.slider("Number of Bedrooms | عدد الغرف", min_value=1, max_value=10, value=1)
bathrooms = st.slider("Number of Bathrooms | عدد الحمامات", min_value=1, max_value=10, value=1)
size = st.number_input("Size in Square Meters | مساحة الشقة م مربع", min_value=0, max_value=1000, value=80)

amenities = st.multiselect(
    "Select Amenities | المرافق",
    options=["Elevator | اسانسير", "Water Meter | عداد مياه", "Natural Gas | غاز طبيعي", "Landline | خط ارضي", "Central AC | تكيف", "Balcony | بلكونة", "Security | امن", "Covered Parking | موقوف السيارات", "Pets Allowed | مسموح بالحيوانات الاليفة", "Private Garden | حديقة خاصة"],
    default=["Water Meter | عداد مياه", "Natural Gas | غاز طبيعي"],
)


predict_rent = st.button("Rent | الإيجار")

if predict_rent:
    # preprocessing inputs
    lat, long = preprocessing.get_coord_lat_lon(full_address)
    
    elevator = int(amenities.__contains__("Elevator | اسانسير"))
    level_cat = preprocessing.categorize_level(level)
    ascore = preprocessing.accessibility_score(level_cat, elevator)
    
    bathtobed = preprocessing.bathtobed_ratio(bathrooms, bedrooms)
    total_rooms = bedrooms + bathrooms

    features = {
        "Size": size, 
        "Bathrooms": bathrooms, 
        "Bedrooms": bedrooms,
        "Lat": lat, 
        "Long": long, 
        "Amenities_Score": len(amenities), 
        "BedtoBath_Ratio": bathtobed, 
        "Furnished": furnished, 
        "Apartment": house_type[0], 
        "Duplex": house_type[1], 
        "Penthouse": house_type[2], 
        "Room": house_type[3], 
        "Studio": house_type[4], 
        "TotalRooms": total_rooms, 
        "Level_category": level_cat, 
        "accessibility_score": ascore, 
    }

    std_feats = scaler_pipeline.transform(pd.DataFrame([features]))
    prediction = model.predict(std_feats)[0]
    price = int(round(prediction, -2))
    st.success(f'### {price:,} EGP/month')
