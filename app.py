import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

st.set_page_config(
    page_title="Diabetic Retinopathy Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("Diabetic Retinopathy Classification")

st.caption("MACHINE LEARNING MODEL EVALUATION")

st.write(
    "An evaluation and comparison of five machine learning classifiers "
    "using the Diabetic Retinopathy Debrecen dataset."
)

st.write(
    "The objective of this dataset is to predict whether retinal images "
    "contain signs of diabetic retinopathy using features extracted from "
    "the Messidor image set."
)

st.markdown("**Models implemented**")

model_col1, model_col2, model_col3, model_col4, model_col5 = st.columns(5)

with model_col1:
    st.markdown("Logistic Regression")

with model_col2:
    st.markdown("Decision Tree")

with model_col3:
    st.markdown("K-Nearest Neighbors")

with model_col4:
    st.markdown("Naive Bayes")

with model_col5:
    st.markdown("Support Vector Machine")

st.divider()

st.subheader("Test Dataset")

uploaded_file = st.file_uploader(
    "Upload another test CSV",
    type=["csv"]
)

if uploaded_file is not None:
    with st.spinner("Loading dataset and updating model results..."):
        data = pd.read_csv(uploaded_file)

    st.success(f"Dataset loaded successfully: {uploaded_file.name}")
    st.caption(f"{len(data)} rows loaded. Model results have been updated.")

else:
    data = pd.read_csv("test_data.csv")
    st.info("Using the default test dataset.")

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", data.shape[0])

with col2:
    st.metric("Total Features", data.shape[1] - 1)

with col3:
    st.metric("Number of Classes", data["Class"].nunique())

st.subheader("Features in the Dataset")

st.write(
    "The dataset contains 19 image-derived features, representing "
    "the following measurements and detection results:"
)

feature_col1, feature_col2 = st.columns(2)

with feature_col1:
    st.markdown("""
    - **Image Quality Assessment (1 feature):** Binary assessment of image quality.
    - **Pre-screening (1 feature):** Identification of severe retinal abnormalities.
    - **Microaneurysm Detection (6 features):** Measurements representing detected microaneurysms at different confidence levels.
    - **Exudate Detection (8 features):** Measurements representing exudate detection, normalized by the region of interest diameter.
    """)

with feature_col2:
    st.markdown("""
    - **Macula-Optic Disc Distance (1 feature):** Distance between the centres of the macula and optic disc, normalized by the region of interest diameter.
    - **Optic Disc Diameter (1 feature):** Measurement of the optic disc.
    - **AM/FM-based Classification (1 feature):** Binary classification result based on the AM/FM method.
    """)

with st.expander("Preview Test Dataset"):
    st.dataframe(data, width="stretch", height=400)

st.divider()

try:
    logistic_model = joblib.load("model/logistic_regression.pkl")
    decision_tree_model = joblib.load("model/decision_tree.pkl")
    knn_model = joblib.load("model/knn.pkl")
    naive_bayes_model = joblib.load("model/naive_bayes.pkl")
    svm_model = joblib.load("model/svm.pkl")
    scaler = joblib.load("model/scaler.pkl")
except Exception as e:
    st.error(f"Could not load the trained models. Error: {e}")
    st.stop()

feature_columns = [str(i) for i in range(19)]
X_test = data[feature_columns]
y_actual = data["Class"]

X_test_scaled = scaler.transform(X_test)

all_models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "K-Nearest Neighbors": knn_model,
    "Naive Bayes": naive_bayes_model,
    "Support Vector Machine": svm_model
}

model_predictions = {}
comparison_results = []

for model_name, model in all_models.items():

    if model_name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]:
        X = X_test_scaled
    else:
        X = X_test

    # Generate predictions
    predictions = model.predict(X)

    model_predictions[model_name] = predictions

    if model_name == "Support Vector Machine":
        y_scores = model.decision_function(X)
    else:
        positive_class_index = list(model.classes_).index(1)
        y_scores = model.predict_proba(X)[:, positive_class_index]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_actual, predictions)
    auc = roc_auc_score(y_actual, y_scores)
    precision = precision_score(y_actual, predictions, zero_division=0)
    recall = recall_score(y_actual, predictions, zero_division=0)
    f1 = f1_score(y_actual, predictions, zero_division=0)
    mcc = matthews_corrcoef(y_actual, predictions)

    comparison_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })


st.subheader("Model Performance Comparison")

comparison_df = pd.DataFrame(comparison_results)

st.dataframe(
    comparison_df.set_index("Model").style.format("{:.4f}"),
    width="stretch"
)

st.write(
    "Logistic Regression achieved the best overall performance across all six evaluation metrics. "
    "Support Vector Machine showed the second-best AUC score, while Decision Tree, "
    "K-Nearest Neighbors, and Naive Bayes achieved comparatively lower overall results."
)

st.subheader("Detailed Model Evaluation")

selected_model = st.selectbox(
    "Select a model to view its detailed evaluation results",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Support Vector Machine"
    ]
)

predictions = model_predictions[selected_model]

selected_results = comparison_df[
    comparison_df["Model"] == selected_model
].iloc[0]

st.markdown(f"**Performance Metrics - {selected_model}**")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{selected_results['Accuracy']:.4f}")

with col2:
    st.metric("AUC Score", f"{selected_results['AUC']:.4f}")

with col3:
    st.metric("Precision", f"{selected_results['Precision']:.4f}")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Recall", f"{selected_results['Recall']:.4f}")

with col2:
    st.metric("F1 Score", f"{selected_results['F1 Score']:.4f}")

with col3:
    st.metric("MCC", f"{selected_results['MCC']:.4f}")

st.divider()

st.subheader(f"Confusion Matrix - {selected_model}")

selected_confusion_matrix = confusion_matrix(y_actual, predictions)

fig, ax = plt.subplots(figsize=(4, 3))

display = ConfusionMatrixDisplay(
    confusion_matrix=selected_confusion_matrix,
    display_labels=["Class 0", "Class 1"]
)

display.plot(
    ax=ax,
    cmap="Blues",
    colorbar=False,
    values_format="d",
    text_kw={"fontsize": 11}
)

fig.tight_layout()
st.pyplot(fig, width=420)
plt.close(fig)

st.subheader(f"Classification Report - {selected_model}")

report = classification_report(
    y_actual,
    predictions,
    target_names=["Class 0", "Class 1"],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.style.format("{:.4f}"),
    width="stretch"
)

st.divider()

st.subheader("Prediction Results")

with st.expander("View Results"):

    correct_predictions = (predictions == data["Class"].values).sum()
    incorrect_predictions = (predictions != data["Class"].values).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Correct Predictions", correct_predictions)

    with col2:
        st.metric("Incorrect Predictions", incorrect_predictions)

    st.subheader("Predicted Class Distribution")

    prediction_counts = pd.Series(predictions).value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(5, 2.5))

    ax.barh(
        ["Class 0", "Class 1"],
        [prediction_counts.get(0, 0), prediction_counts.get(1, 0)]
    )

    ax.set_xlabel("Number of Predictions")
    ax.set_ylabel("Predicted Class")
    ax.set_title("Predicted Class Distribution")

    fig.tight_layout()
    st.pyplot(fig, width=420)
    plt.close(fig)

    st.subheader("Dataset with Predictions")

    prediction_details = data.copy()
    prediction_details["Predicted_Class"] = predictions

    st.dataframe(prediction_details, width="stretch", height=400)
