import pickle
import streamlit as st
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer outside of the button callbacks to persist
if 'model' not in st.session_state:
    with open('bullying_model_sgd.pkl', 'rb') as f:
        st.session_state.model = pickle.load(f)
    with open('BullyingVectorizer_sgd.pkl', 'rb') as f:
        st.session_state.vectorizer = pickle.load(f)

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
        # Transform input text to model-compatible format
        X_new = st.session_state.vectorizer.transform([user_input])

        # Predict using the model
        predicted = st.session_state.model.predict(X_new)[0]
        if predicted == 0:
            predicted = 'Not Bullying'
        if predicted == 1:
            predicted = 'Bullying'
        # Update predicted emotion in session state
        st.session_state.predicted_emotion = predicted

# Show prediction if available
if st.session_state.predicted_emotion is not None:
    st.success(f"Predicted Emotion: **{st.session_state.predicted_emotion}**")

# Dropdown to update label
label = st.selectbox("✅ Confirm or correct the emotion label:", classes)

# Update Model Button
if st.button("📈 Update Model"):
    if user_input.strip():
        # Transform input to vectorized form
        X_new = st.session_state.vectorizer.transform([user_input])

        # Prediction before model update
        before = st.session_state.model.predict(X_new)[0]
        st.write(f"Prediction before update: {before}")

        # First-time setup for partial_fit (if model doesn't have 'classes_')
        if not hasattr(st.session_state.model, 'classes_'):
            st.session_state.model.partial_fit(X_new, [label], classes=classes)
        else:
            st.session_state.model.partial_fit(X_new, [label])

        # Prediction after model update
        after = st.session_state.model.predict(X_new)[0]
        st.write(f"Prediction after update: {after}")

        # Save updated model and vectorizer
        with open('bullying_model_sgd.pkl', 'wb') as f:
            pickle.dump(st.session_state.model, f)
        with open('BullyingVectorizer_sgd.pkl', 'wb') as f:
            pickle.dump(st.session_state.vectorizer, f)

        # Show update info
        st.info(f"🔄 Model Updated\n**Before:** {before}\n**After:** {after}")

        # Clear the old prediction
        st.session_state.predicted_emotion = ""  

# Save input for next run
st.session_state.user_input = user_input
