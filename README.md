# Hand Gesture Classification from Text

This project is a machine learning application that classifies text descriptions into corresponding hand gestures. The application is built with Scikit-learn and deployed as an interactive web app using Streamlit.

## Table of Contents
- [How it Works](#how-it-works)
- [How to Use](#how-to-use)
- [How to Collaborate](#how-to-collaborate)
- [License](#license)

## How it Works

The project uses a Natural Language Processing (NLP) pipeline to classify text. Here's a breakdown of the process:

1.  **Text Preprocessing:** The input text is preprocessed using the NLTK library. This involves tokenization and lemmatization to normalize the text.
2.  **TF-IDF Vectorization:** The preprocessed text is then converted into a numerical representation using a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer.
3.  **Classification:** A trained Linear Support Vector Classifier (LinearSVC) model, which has been calibrated using `CalibratedClassifierCV` to provide prediction probabilities, classifies the vectorized text into one of the following hand gestures:
    *   `pointing`
    *   `good sign`
    *   `hello sign`
4.  **Label Encoding:** The predicted numerical label is converted back to its human-readable form (e.g., 'good sign').

The trained models (`tfidf_vectorizer.joblib`, `calibrated_clf.joblib`, and `encoder.joblib`) are included in this repository.

## How to Use

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Hand_Gesture_classification
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    The model uses the 'wordnet' corpus from NLTK. Run the following command in a Python interpreter to download it:
    ```python
    import nltk
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('omw-1.4')
    ```

5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    This will open the web application in your browser.

## How to Collaborate

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/your-feature-name`).
5.  Open a pull request.

Please make sure to update tests as appropriate.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
