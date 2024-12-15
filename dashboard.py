import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model and label encoder
@st.cache_resource
def load_model_and_encoder():
    try:
        model = load('sentiment_model.pkl')  # Replace with your model file name
        label_encoder = load('label_encoder.pkl')  # Replace with your LabelEncoder file name
        return model, label_encoder
    except Exception as e:
        st.error(f"Error loading model or encoder: {e}")
        return None, None

# Predict sentiment for single text
def predict_sentiment(model, label_encoder, text):
    prediction_encoded = model.predict([text])[0]
    sentiment = label_encoder.inverse_transform([prediction_encoded])[0]  # Decode sentiment
    confidence = max(model.predict_proba([text])[0])
    return sentiment, confidence

# Sentiment-to-color mapping
def sentiment_to_color(sentiment):
    colors = {
        'positive': '#28a745',  # Green
        'negative': '#dc3545',  # Red
        'neutral': '#6c757d'   # Gray
    }
    return colors.get(sentiment, '#007bff')  # Default: Blue

# Bar chart for sentiment distribution
def plot_sentiment_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(
        x='predicted_sentiment',
        data=df,
        palette=['#28a745', '#dc3545', '#6c757d']
    )
    plt.title('Sentiment Distribution', fontsize=18, fontweight='bold')
    plt.xlabel('Sentiment', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(plt)

# Dashboard layout
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Amo's Sentiment Analysis Dashboard",
        layout="wide",
        page_icon="💬"
    )

    # Custom CSS for cozy mode
    st.markdown(
        """
        <style>
        .stApp {
            font-family: "Georgia", serif;
            background-color: #fefbe9;
            color: #6b4226;
            padding: 10px;
        }
        .block-container {
            max-width: 1200px;
            margin: auto;
            background-color: #fff5e1;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        h1 {
            color: #b3541e;
            font-size: 36px;
            font-weight: bold;
        }
        h2 {
            color: #b3541e;
        }
        .stButton>button {
            background-color: #b3541e;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.title("💬 Amo's Sentiment Analysis Dashboard")
    st.write("Analyze text sentiment with ease! Use the options below to analyze individual texts or bulk data from a CSV file.")

    # Load model and encoder
    model, label_encoder = load_model_and_encoder()
    if not model or not label_encoder:
        st.warning("⚠️ No model or encoder found. Please ensure you have the necessary files.")
        return

    # Create a tab layout
    tabs = st.tabs(["📄 Single Text Analysis", "📊 Batch Analysis"])

    # Single Text Analysis
    with tabs[0]:
        st.header("📄 Single Text Analysis")
        user_input = st.text_area("Enter text for analysis:", height=120, placeholder="Type something here...")
        if user_input:
            sentiment, confidence = predict_sentiment(model, label_encoder, user_input)
            color = sentiment_to_color(sentiment)

            # Display sentiment result
            st.markdown(
                f"<h3 style='color:{color}'>{sentiment.capitalize()} ({confidence:.2%} confident)</h3>",
                unsafe_allow_html=True
            )
            st.success(f"The sentiment for your text is **{sentiment.capitalize()}** with a confidence of **{confidence:.2%}**.")

    # Batch Analysis
    with tabs[1]:
        st.header("📊 Batch Analysis")
        uploaded_file = st.file_uploader("Upload a CSV file for batch analysis:", type=["csv"])
        if uploaded_file:
            try:
                data = pd.read_csv(uploaded_file)
                if 'processed_text' not in data.columns:
                    st.error("The uploaded file must contain a 'processed_text' column.")
                    return

                # Batch predictions
                data['predicted_sentiment_encoded'] = data['processed_text'].apply(
                    lambda x: model.predict([x])[0]
                )
                data['predicted_sentiment'] = data['predicted_sentiment_encoded'].apply(
                    lambda x: label_encoder.inverse_transform([x])[0]
                )

                # Display results
                st.success(f"Processed {len(data)} rows successfully!")
                st.dataframe(data[['processed_text', 'predicted_sentiment']].head())

                # Plot sentiment distribution
                plot_sentiment_distribution(data)

                # Download predictions
                st.download_button(
                    "📥 Download Predictions",
                    data.to_csv(index=False),
                    file_name="predictions.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
