# Fake_News_Detection

# Fake News Detection using Machine Learning

This project uses natural language processing (NLP) and machine learning techniques to classify news articles as "Real" (Not Fake) or "Fake". The workflow involves loading and preprocessing text data, converting text into numerical vectors using TF-IDF, and training several classification models to compare their performance.

## Dataset

The project relies on two separate CSV files:
* `Fake.csv`: Contains news articles that are classified as fake.
* `True.csv`: Contains legitimate news articles.

Each dataset includes columns such as `title`, `text`, `subject`, and `date`. For this project, only the `text` column is used for classification, along with a target `class` label.

## Project Workflow

1.  **Data Loading:** The `Fake.csv` and `True.csv` files are loaded into pandas DataFrames.
2.  **Labeling:** A `class` column is added.
    * `0` for articles from `Fake.csv`
    * `1` for articles from `True.csv`
3.  **Data Splicing:** The last 10 rows from each file are separated and reserved for a final manual testing demonstration.
4.  **Merging and Shuffling:** The remaining data from both files is merged into a single DataFrame. This combined dataset is then shuffled to ensure the models do not learn from the order of the data.
5.  **Feature Selection:** Unnecessary columns (`title`, `date`, `subject`) are dropped, leaving only the `text` and `class` columns.
6.  **Text Preprocessing:** A custom function (`wordopt`) is applied to the `text` column to clean the data. This function:
    * Converts text to lowercase.
    * Removes URLs and website links.
    * Removes HTML tags.
    * Removes text in square brackets (e.g., `[Reuters]`).
    * Removes all punctuation.
    * Removes words containing numbers.
7.  **Vectorization:** The cleaned text data is converted into a numerical format using `TfidfVectorizer`, which reflects the importance of a word in a document relative to its frequency in the entire corpus.
8.  **Train-Test Split:** The dataset is split into training (75%) and testing (25%) sets.

---

## Model Training and Evaluation

Four different machine learning models are trained and evaluated on the test set.

1.  **Logistic Regression**
    * **Accuracy:** ~98.6%
    * **Classification Report:**
        ```
              precision    recall  f1-score   support
           0       0.99      0.98      0.99      5906
           1       0.98      0.99      0.99      5314
        ```

2.  **Decision Tree Classifier**
    * **Accuracy:** ~99.4%
    * **Classification Report:**
        ```
              precision    recall  f1-score   support
           0       0.99      1.00      0.99      5906
           1       1.00      0.99      0.99      5314
        ```

3.  **Gradient Boosting Classifier**
    * **Accuracy:** ~99.5%
    * **Classification Report:**
        ```
              precision    recall  f1-score   support
           0       1.00      0.99      1.00      5906
           1       0.99      1.00      0.99      5314
        ```

4.  **Random Forest Classifier**
    * **Accuracy:** ~99.0%
    * **Classification Report:**
        ```
              precision    recall  f1-score   support
           0       1.00      0.99      1.00      5906
           1       0.99      1.00      0.99      5314
        ```

---

## How to Use

### 1. Run the Notebook
You can run the `Fake news detection .ipynb` notebook sequentially to load the data, train the models, and see the evaluation metrics.

### 2. Manual Testing
The notebook includes a `manual_testing` function that allows you to input your own news text and get a prediction from all four trained models.

To use it, run the final cell and paste a news article into the input prompt.

**Example Output:**


[OUTPUT] LR prediction: Not a Fake news DT prediction : Fake News GBC prediction: Fake News RFC prediction :Fake News


---

## Requirements

You can install the necessary Python libraries using pip:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn

Libraries Used:

pandas

numpy

seaborn

matplotlib

scikit-learn

re (built-in)

string (built-in)
