import streamlit as st
import pickle

# --- Load model and vectorizer only once and keep them in session state ---
if "model" not in st.session_state:
    with open('bullying_model_svm_two.pkl', 'rb') as f:
        st.session_state.model = pickle.load(f)
    with open('BullyingVectorizer_svm_two.pkl', 'rb') as f:
        st.session_state.vectorizer = pickle.load(f)

if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "predicted_emotion" not in st.session_state:
    st.session_state.predicted_emotion = ""

model = st.session_state.model
vectorizer = st.session_state.vectorizer
classes = [0, 1]
bullying_dict = {0: "Not Bullying", 1: "Bullying"}

st.title("💬 Bullying Detection in Transport & Online Model Update")

# User input
user_input = st.text_input(
    "✏️ Enter a sentence to detect whether it is bullying:",
    value=st.session_state.user_input
)

predict_clicked = st.button("🔍 Predict If It Is Bullying")

if predict_clicked:
    if user_input.strip():
        X_new = vectorizer.transform([user_input])
        predicted_bullying = model.predict(X_new)[0]
        st.session_state.predicted_emotion = predicted_bullying
        st.session_state.user_input = user_input  # Save current input

# --- Display prediction ---
if st.session_state.predicted_emotion != "":
    st.success(f"Prediction: **{bullying_dict[st.session_state.predicted_emotion]}**")

# --- Label correction ---
label = st.selectbox("✅ Confirm or correct the bullying label:", classes)

# --- Update model button ---
update_clicked = st.button("📈 Update Model")

if update_clicked:
    if user_input.strip():
        # Convert input into vector form
        X_new = vectorizer.transform([user_input])

        # Print before model prediction (for debugging)
        before = model.predict(X_new)[0]
        st.write(f"Before update prediction: {bullying_dict[before]}")

        # Update the model with new label using partial_fit
        try:
            # Using partial_fit to update model with the new input and label
            if not hasattr(model, 'classes_'):
                st.write("Classes_ attribute is not present, passing classes explicitly.")
                model.partial_fit(X_new, [label], classes=classes)
            else:
                st.write("Classes_ attribute is present, updating model.")
                model.partial_fit(X_new, [label])
        except Exception as e:
            st.error(f"Error while updating model: {e}")
            return

        # Print after model prediction (for debugging)
        after = model.predict(X_new)[0]
        st.write(f"After update prediction: {bullying_dict[after]}")

        # Save updated model and vectorizer to disk
        with open('bullying_model_svm_two.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('BullyingVectorizer_svm_two.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)

        # Save the updated model and vectorizer directly into session state
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer

        # Provide feedback to user
        st.info(f"🔄 Model Updated\n**Before:** {bullying_dict[before]}\n**After:** {bullying_dict[after]}")

        # Reset after update
        st.session_state.predicted_emotion = ""
        st.session_state.user_input = ""
