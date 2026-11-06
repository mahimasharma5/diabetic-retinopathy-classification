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
    layout="wide"
)

st.title("Diabetic Retinopathy Classification")
st.write("Performance of 5 ML classification models on the diabetic retinopathy debrecen dataset.",
         "ML Models implemented:",
         "1. Logistic Regression",
         "2. Decision Tree Classifier",
         "3. K-Nearest Neighbors",
         "4. Gaussian Naive Bayes",
         "5. Support Vector Machine")

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader("Choose your test CSV file", 
                                 type=["csv"])


if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error during csv upload: {e}")
        st.stop()
    st.success("Dataset uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(data.head(5), width="stretch")

    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", data.shape[0])
    with col2:
        st.metric("Total Columns", data.shape[1])

    st.subheader("Loading trained models")
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
    st.success("All five trained models and the scaler loaded successfully!")

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

    st.subheader("Comparison of All Five Models")

    comparison_df = pd.DataFrame(comparison_results)
    st.dataframe(comparison_df.set_index("Model"), width="stretch")

    st.subheader("Select a Classification Model")
    selected_model = st.selectbox("Choose a model to view its detailed evaluation results",
                                  ["Logistic Regression",
                                   "Decision Tree",
                                   "K-Nearest Neighbors",
                                   "Naive Bayes",
                                   "Support Vector Machine"])

    predictions = model_predictions[selected_model]

    st.subheader(f"Confusion Matrix - {selected_model}")

    selected_confusion_matrix = confusion_matrix(y_actual, predictions)
    fig, ax = plt.subplots(figsize=(4, 3))
    display = ConfusionMatrixDisplay(confusion_matrix=selected_confusion_matrix, display_labels=["Class 0", "Class 1"])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d", text_kw={"fontsize": 11})
    fig.tight_layout()
    st.pyplot(fig, width=420)
    plt.close(fig)

    st.subheader(f"Classification Report - {selected_model}")

    report = classification_report(y_actual,
                                   predictions,
                                   target_names=["Class 0", "Class 1"],
                                   output_dict=True,
                                   zero_division=0)

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df, width="stretch")

    st.subheader("Prediction Results")
    if st.button("Generate Predictions"):
        prediction_results = data.head(5).copy()
        prediction_results["Predicted_Class"] = predictions[:5]
        st.success(f"First five predictions generated for {selected_model}!")

        st.dataframe(prediction_results, width="stretch")
        st.subheader("Predicted Class Distribution")
        prediction_counts = (pd.Series(predictions).value_counts().sort_index())

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Class 0", prediction_counts.get(0, 0))

        with col2:
            st.metric("Class 1", prediction_counts.get(1, 0))
else:
    st.info("Please upload the test_data.csv file.")