import streamlit as st
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
import pickle

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'bullying_prediction' not in st.session_state:
    st.session_state.predicted_emotion = ""
    
with open('bullying_model_svm.pkl', 'rb') as f:
    model = pickle.load(f)
with open('BullyingVectorizer_svm.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
classes = [0,1]


st.title("💬 Bullying Detection in Transport & Online Model Update")

user_input = st.text_input("✏️ Enter a sentence to detect whether it is bullying:", value=st.session_state.user_input)

if st.button("🔍 Predict If It Is Bullying"):
    if user_input.strip():
        X_new = vectorizer.transform([user_input])
        predicted_bullying = model.predict(X_new)[0]
        st.session_state.predicted_emotion = predicted_bullying

if st.session_state.predicted_emotion:
    st.success(f"Predicted Emotion: **{st.session_state.predicted_emotion}**")

label = st.selectbox("✅ Confirm or correct the bullying label:", classes)

if st.button("📈 Update Model"):
    if user_input.strip():
        X_new = vectorizer.transform([user_input])
        before = model.predict(X_new)[0]
        if not hasattr(model, 'classes_'):
            model.partial_fit(X_new, [label], classes=classes)
        else:
            model.partial_fit(X_new, [label])
        after = model.predict(X_new)[0]

        with open('bullying_model_svm.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('BullyingVectorizer_svm.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        st.info(f"🔄 Model Updated\n**Before:** {before}\n**After:** {after}")
        st.session_state.predicted_emotion = "" 

st.session_state.user_input = user_input
