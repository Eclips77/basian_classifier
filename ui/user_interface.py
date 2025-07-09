import streamlit as st
from src.app_controller import AppController
from services.file_loader import FileLoader

# Initialize once
loader = FileLoader()
controller = AppController("path/to/your_file.csv", "label_column_name", loader)

st.set_page_config(page_title="Naive Bayes Classifier", layout="centered")
st.title("🧮 Naive Bayes Classifier App")

# 1️⃣ Load and Clean Data
if st.button("1️⃣ Load and Clean Data"):
    controller.load_and_prepare()
    st.success("✅ Data loaded and cleaned successfully!")

# 2️⃣ Train Model
if st.button("2️⃣ Train Model"):
    if controller.X_train is None:
        st.error("⚠️ Please load and clean data first.")
    else:
        controller.train_model()
        st.success("✅ Model trained successfully!")

# 3️⃣ Evaluate Model Accuracy
if st.button("3️⃣ Evaluate Model Accuracy"):
    if controller.evaluator is None:
        st.error("⚠️ Please train the model first.")
    else:
        accuracy = controller.evaluate_model()
        st.write(f"📊 **Model Accuracy:** `{accuracy:.2%}`")

# 4️⃣ Predict a Single Record
st.subheader("🔍 Predict a Single Record")

if controller.X_test is not None and controller.evaluator is not None:
    record = {}
    for col in controller.X_test.columns:
        record[col] = st.text_input(f"Enter value for `{col}`")

    if st.button("🔮 Predict"):
        if any(v.strip() == "" for v in record.values()):
            st.warning("⚠️ Please fill in all fields.")
        else:
            result = controller.predict_record(record)
            if result:
                st.success(f"✅ **Prediction:** `{result}`")
else:
    st.info("ℹ️ Load data and train the model to enable prediction.")
