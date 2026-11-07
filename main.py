import streamlit as st
import tensorflow as tf
import numpy as np


USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

def login():
    st.title("🔐 Login - Plant Disease Detection System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid username or password")


def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    result_index = np.argmax(predictions)
    return result_index


def main_app():
   
    st.sidebar.title("Dashboard")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])
    
    if app_mode == "Home":
        st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
        st.image("home_page.jpeg", use_container_width=True)
        st.markdown("""
        Welcome to the Plant Disease Recognition System! 🌿🔍
        
        Our mission is to help in identifying plant diseases efficiently.
        Upload an image of a plant, and our system will analyze it to detect any signs of diseases.
        Together, let's protect our crops and ensure a healthier harvest!
        
        ### How It Works
        1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
        2. **Analysis:** Our system processes the image using advanced algorithms to identify potential diseases.
        3. **Results:** View the results along with recommendations for further action.
        
        ### Why Choose Us?
        - **Accuracy:** Utilizes state-of-the-art machine learning techniques.
        - **User-Friendly:** Simple and intuitive interface.
        - **Efficiency:** Delivers fast results, aiding quick decision-making.
        
        Click on the **Disease Recognition** page in the sidebar to get started!
        """)
    
    elif app_mode == "About":
        st.header("About")
        st.markdown("""
        #### About Dataset
        This dataset was recreated using offline augmentation from the original dataset available on a GitHub repo.
        It consists of about 87K RGB images of healthy and diseased crop leaves categorized into 38 classes.
        The dataset is split in an 80/20 ratio for training and validation, with an additional directory containing 33 test images.
        
        #### Content
        1. train (70,295 images)
        2. test (33 images)
        3. validation (17,572 images)
        """)
    
    elif app_mode == "Disease Recognition":
        st.header("Disease Recognition")
        test_image = st.file_uploader("Choose an Image:")
        if st.button("Show Image") and test_image is not None:
            st.image(test_image, use_container_width=True)
        if st.button("Predict") and test_image is not None:
            st.snow()
            st.write("Our Prediction...")
            result_index = model_prediction(test_image)
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            prediction_label = class_name[result_index] 

            if "healthy" in prediction_label.lower():
                st.success("✅ The plant is healthy.")
            else:
                st.warning(f"⚠️ The plant has **{prediction_label.replace('_', ' ')}**.")
            

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:
    login()
else:
    main_app()
