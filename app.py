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
        # Debug: Display user input
        

        # Transform input text to model-compatible format
        X_new = st.session_state.vectorizer.transform([user_input])

        # Debug: Check the transformed input
        

        # Predict using the model
        predicted = st.session_state.model.predict(X_new)[0]
        if predicted == 0:
            predicted = 'Not Bullying'
        if predicted == 1:
            predicted = 'Bullying'
        # Debug: Show prediction before model update
        

        # Update predicted emotion in session state
        st.session_state.predicted_emotion = predicted

# Show prediction if available
if st.session_state.predicted_emotion is not None:
    st.success(f"Predicted Emotion: **{st.session_state.predicted_emotion}**")

# Dropdown to update label
label = st.selectbox("✅ Confirm or correct the emotion label:", classes)

# Debug: Show selected label
st.write(f"Selected Label: {label}")

# Update Model Button
if st.button("📈 Update Model"):
    if user_input.strip():
        # Debug: Display user input and selected label
        

        # Transform input to vectorized form
        X_new = st.session_state.vectorizer.transform([user_input])

        # Prediction before model update
        before = st.session_state.model.predict(X_new)[0]
        

        # First-time setup for partial_fit (if model doesn't have 'classes_')
        if not hasattr(st.session_state.model, 'classes_'):
            st.write("Model does not have 'classes_' attribute, initializing 'partial_fit'.")
            for i in range(10):
                st.session_state.model.partial_fit(X_new, [label], classes=classes)
        else:
            st.write(f"Model has 'classes_' attribute, updating with 'partial_fit'.")
            for i in range(10):
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

html_code="""
<style>
body {
background-color:black;
}
h1 {
font-family: Courier New;
}
</style>
<h1>What is Bullying in the First Place?</h1>
<p>Bullying in Transit: A Growing Concern in Modern Transportation
Introduction
Bullying is a persistent social issue that transcends the boundaries of age, setting, and circumstance. One often-overlooked context where bullying occurs is during transportation—whether it's the daily school commute or public transit in urban settings. Despite safety measures like security cameras, transportation staff, and law enforcement presence, bullying during transit remains a pressing issue. It manifests in both physical and digital forms, affecting individuals emotionally, psychologically, and, in some cases, physically.

Defining Bullying
According to the Oxford English Dictionary, bullying is “the use of strength or influence to intimidate or harm someone who is perceived as vulnerable.” More broadly, it refers to repeated, intentional actions aimed at hurting, threatening, or demeaning another individual. These actions can be verbal, physical, psychological, or virtual. While bullying is commonly associated with school environments, it also occurs in everyday public spaces—including transit systems.

Environments Where Bullying Occurs
Bullying can arise in virtually any environment: schools, workplaces, online platforms, and public areas such as markets and buses. This article focuses specifically on bullying within the context of transportation, both among school-aged children and in the broader community. The limited supervision and close quarters of transit settings often create opportunities for unchecked behavior.

Types of Bullying in Transit Settings
Two dominant forms of bullying are frequently observed during transportation:
Physical Bullying
 Involves direct bodily harm or intimidation, including pushing, shoving, hitting, or the use of offensive gestures or language. On school buses or public transit, these actions may go unnoticed due to crowding or distractions.


Cyberbullying
 Refers to the use of digital technology—such as smartphones or social media—to harass, shame, or intimidate someone. This could involve taking unauthorized photos or videos of individuals and sharing them publicly or within private groups.



Bullying on School Transportation
School buses, despite being supervised, are common spaces for bullying. Children may be teased, ridiculed, excluded, or physically harmed by their peers. In some cases, students are filmed without their consent, and these recordings are shared online to further humiliate them. The emotional impact of such incidents can be severe, leading to anxiety, isolation, or depression.
In response, school authorities often impose disciplinary actions such as detention, parental notifications, or suspension. However, prevention and early intervention are more effective than reactive measures.

Addressing School-Based Transit Bullying
To mitigate bullying during school transportation, the following steps are recommended:
Immediate Reporting: Students should be encouraged to report incidents to the bus driver, aide, or school staff immediately.


Peer Support: Witnesses of bullying should intervene in a safe manner and support the victim by speaking up or notifying an adult.


Digital Responsibility: Any evidence of cyberbullying should be shown to a trusted adult or school official, who can then escalate it appropriately.



Bullying in Public Transit: A Broader Perspective
In public transportation systems, adults may also experience bullying in the form of verbal abuse, physical intimidation, or inappropriate behavior. Unlike in schools, adult bullying is often minimized or misinterpreted as rudeness or conflict. In reality, it can deeply impact a person's mental well-being and sense of public safety.
Cyberbullying in public transit may occur through unauthorized recordings that are shared online to mock or defame someone. The public nature of such content can make it especially distressing.

Responding to Transit Bullying in Public Spaces
Effective responses to bullying in public transit include:
Notifying Authorities: Victims should report incidents to transit officials or law enforcement, depending on severity.


Raising Awareness: Encouraging bystanders to speak up or report observed incidents can help create a culture of accountability.


Documenting Evidence: When safe to do so, collecting evidence (like screenshots or video) can aid investigations and discourage further behavior.


Bullying—whether physical or digital—remains a significant challenge in both school and public transportation settings. It affects individuals of all ages and can have long-lasting consequences on emotional and psychological health. Addressing this issue requires a collective effort from peers, educators, transit staff, and law enforcement. Through awareness, intervention, and continued advocacy, we can create safer, more respectful transportation environments for everyone.
</p>
<img src="https://www.stopbullying.gov/sites/default/files/2019-11/sb-620x529-laws_policy_map.gif" alt="This is a graph of the which U.S. states have laws or policies or none of the above to prevent bullying." width="600" height="500">
<h3>So this is where you will write the comment that bullyer said and find out whoever is bullying to work.</h3>"""
st.markdown(html_code, unsafe_allow_html=True)

