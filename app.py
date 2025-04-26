import pickle
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
with open('bullying_model_sgd.pkl', 'rb') as f:
    model = pickle.load(f)
with open('BullyingVectorizer_sgd.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Class labels for training
classes = [0, 1]

# Session state for persistent UI elements
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'predicted_emotion' not in st.session_state:
    st.session_state.predicted_emotion = ""

st.title("💬 Bullying prediction & Online Model Update")

# Text input
user_input = st.text_input("✏️ Enter a sentence:", value=st.session_state.user_input)

# Predict Emotion Button
if st.button("🔍 Predict Emotion"):
    if user_input.strip():
        # Debug: Check if input is valid
        st.write(f"User Input: {user_input}")
        
        # Transform input text to model-compatible format
        X_new = vectorizer.transform([user_input])
        
        # Debug: Check the transformed input
        st.write(f"Transformed Input: {X_new}")
        
        # Predict using the model
        predicted = model.predict(X_new)[0]
        
        # Debug: Check prediction
        st.write(f"Prediction: {predicted}")
        
        # Update predicted emotion in session state
        st.session_state.predicted_emotion = predicted

# Show prediction if available
if st.session_state.predicted_emotion:
    st.success(f"Predicted Emotion: **{st.session_state.predicted_emotion}**")

# Dropdown to update label
label = st.selectbox("✅ Confirm or correct the emotion label:", classes)

# Update Model Button
if st.button("📈 Update Model"):
    if user_input.strip():
        # Debug: Display user input and selected label
        st.write(f"User Input for Model Update: {user_input}")
        st.write(f"Selected Label for Update: {label}")
        
        # Transform input to vectorized form
        X_new = vectorizer.transform([user_input])

        # Prediction before model update
        before = model.predict(X_new)[0]
        st.write(f"Prediction before update: {before}")
        
        # First-time setup for partial_fit (if model doesn't have 'classes_')
        if not hasattr(model, 'classes_'):
            st.write("Model does not have 'classes_' attribute, initializing 'partial_fit'.")
            model.partial_fit(X_new, [label], classes=classes)
        else:
            st.write(f"Model has 'classes_' attribute, updating with 'partial_fit'.")
            model.partial_fit(X_new, [label])
        
        # Prediction after model update
        after = model.predict(X_new)[0]
        st.write(f"Prediction after update: {after}")

        # Save updated model and vectorizer
        with open('bullying_model_sgd.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('BullyingVectorizer_sgd.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        # Show update info
        st.info(f"🔄 Model Updated\n**Before:** {before}\n**After:** {after}")

        # Clear the old prediction
        st.session_state.predicted_emotion = ""  

# Save input for next run
st.session_state.user_input = user_input
