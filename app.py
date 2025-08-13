import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# ===========================
# Load Data & Model
# ===========================
df = pd.read_csv(r"C:\Users\Hp\OneDrive\Desktop\machine learning-project\data\Titanic.csv")
model = pickle.load(open("model.pkl", "rb"))

# ===========================
# Sidebar Navigation
# ===========================
st.sidebar.title("📌 Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Data Exploration", "📈 Visualisation", "🔮 Prediction", "📏 Model Performance"]
)

# ===========================
# 1. Home Page
# ===========================
if menu == "🏠 Home":
    st.title("🚢 Titanic Survival Prediction App")
    st.write("""
    This app predicts whether a Titanic passenger survived or not,  
    based on input features like age, class, and fare.
    - **Dataset:** Titanic dataset (Kaggle)
    - **Model:** Logistic Regression
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/f/fd/RMS_Titanic_3.jpg", use_column_width=True)

# ===========================
# 2. Data Exploration
# ===========================
elif menu == "📊 Data Exploration":
    st.header("Dataset Overview")
    st.write("Shape of dataset:", df.shape)
    st.write("Columns:", list(df.columns))
    st.write("Data types:", df.dtypes)
    
    st.subheader("Sample Data")
    st.write(df.head())

    # Interactive Filter
    pclass_filter = st.selectbox("Select Passenger Class:", [1, 2, 3])
    st.write(df[df["Pclass"] == pclass_filter])

# ===========================
# 3. Visualisation
# ===========================
elif menu == "📈 Visualisation":
    st.header("Data Visualisations")

    # Chart 1: Survival Count
    fig, ax = plt.subplots()
    sns.countplot(x="Survived", data=df, ax=ax)
    st.pyplot(fig)

    # Chart 2: Age distribution
    fig2, ax2 = plt.subplots()
    sns.histplot(df["Age"], bins=20, kde=True, ax=ax2)
    st.pyplot(fig2)

    # Chart 3: Fare by Class
    fig3, ax3 = plt.subplots()
    sns.boxplot(x="Pclass", y="Fare", data=df, ax=ax3)
    st.pyplot(fig3)

# ===========================
# 4. Prediction
# ===========================
elif menu == "🔮 Prediction":
    st.header("Make a Prediction")

    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.slider("Age", 0, 80, 25)
    sibsp = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.0)
    sex_male = st.selectbox("Sex", ["Male", "Female"])
    embarked_q = st.selectbox("Embarked at Q?", ["No", "Yes"])
    embarked_s = st.selectbox("Embarked at S?", ["No", "Yes"])

    # Convert categorical inputs
    sex_male_val = 1 if sex_male == "Male" else 0
    embarked_q_val = 1 if embarked_q == "Yes" else 0
    embarked_s_val = 1 if embarked_s == "Yes" else 0

    # Create input data
    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Sex_male': [sex_male_val],
        'Embarked_Q': [embarked_q_val],
        'Embarked_S': [embarked_s_val]
    })

    if st.button("Predict"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[0][1]  # Survival probability

        if prediction[0] == 1:
            st.success(f"✅ Passenger would have SURVIVED (Confidence: {prediction_proba:.2f})")
        else:
            st.error(f"❌ Passenger would NOT have survived (Confidence: {1 - prediction_proba:.2f})")

# ===========================
# 5. Model Performance
# ===========================
elif menu == "📏 Model Performance":
    st.header("Model Evaluation Metrics")

    # Assuming you have X_test, y_test saved
    # For demonstration, let's split from df
    from sklearn.model_selection import train_test_split

    feature_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male', 'Embarked_Q', 'Embarked_S']
    X = df[feature_cols]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = model.predict(X_test)

    st.write("**Accuracy:**", accuracy_score(y_test, y_pred))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax_cm)
    st.pyplot(fig_cm)


