import os
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_nlp
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

# Set backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

st.set_page_config(page_title="Disaster Tweet Classifier", layout="wide")
st.title("🌪️ Disaster Tweet Classifier using DistilBERT")

# File upload section
st.sidebar.header("Upload CSV Files")
train_file = st.sidebar.file_uploader("Upload train.csv", type="csv")
test_file = st.sidebar.file_uploader("Upload test.csv", type="csv")
sample_file = st.sidebar.file_uploader("Upload sample_submission.csv", type="csv")

if train_file and test_file and sample_file:
    with st.spinner("Reading files..."):
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        sample_submission = pd.read_csv(sample_file)

        df_train["length"] = df_train["text"].apply(len)
        df_test["length"] = df_test["text"].apply(len)

        st.subheader("📊 Dataset Info")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Set**")
            st.write(df_train.describe())
        with col2:
            st.markdown("**Test Set**")
            st.write(df_test.describe())

        # Model Config
        BATCH_SIZE = 32
        EPOCHS = 2
        VAL_SPLIT = 0.2
        preset = "distil_bert_base_en_uncased"

        # Train/Val Split
        X = df_train["text"]
        y = df_train["target"]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VAL_SPLIT, random_state=42)
        X_test = df_test["text"]

        # Preprocessor and Classifier
        with st.spinner("Loading DistilBERT model..."):
            preprocessor = keras_nlp.models.DistilBertPreprocessor.from_preset(
                preset, sequence_length=160, name="preprocessor_tweets"
            )

            classifier = keras_nlp.models.DistilBertClassifier.from_preset(
                preset, preprocessor=preprocessor, num_classes=2
            )

            classifier.compile(
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer="adam",
                metrics=["accuracy"]
            )

        with st.spinner("Training model..."):
            history = classifier.fit(
                x=X_train,
                y=y_train,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                validation_data=(X_val, y_val),
                verbose=0
            )
        st.success("✅ Model training complete!")

        # Plotting
        st.subheader("📈 Training History")
        fig, ax = plt.subplots()
        ax.plot(history.history["accuracy"], label="Train Acc")
        ax.plot(history.history["val_accuracy"], label="Val Acc")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy")
        ax.set_title("Training vs Validation Accuracy")
        ax.legend()
        st.pyplot(fig)

        # Confusion Matrix
        def show_confusion_matrix(y_true, y_pred, title):
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay.from_predictions(
                y_true,
                np.argmax(y_pred, axis=1),
                display_labels=["Not Disaster", "Disaster"],
                cmap=plt.cm.Blues,
                ax=ax
            )
            tn, fp, fn, tp = confusion_matrix(y_true, np.argmax(y_pred, axis=1)).ravel()
            f1_score = tp / (tp + ((fn + fp) / 2))
            ax.set_title(f"{title} — F1 Score: {f1_score:.2f}")
            st.pyplot(fig)

        y_pred_train = classifier.predict(X_train)
        y_pred_val = classifier.predict(X_val)

        st.subheader("📌 Confusion Matrices")
        st.markdown("**Training Dataset**")
        show_confusion_matrix(y_train, y_pred_train, "Training")

        st.markdown("**Validation Dataset**")
        show_confusion_matrix(y_val, y_pred_val, "Validation")

        # Predict and export
        st.subheader("📤 Generate Submission")
        sample_submission["target"] = np.argmax(classifier.predict(X_test), axis=1)

        csv = sample_submission.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download submission.csv", csv, "submission.csv", "text/csv")

else:
    st.warning("Please upload all required CSV files to begin.")
