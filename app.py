import nltk
import os
import pandas as pd
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load the CSV data into a Pandas DataFrame
data = pd.read_csv('corpus.csv')

# Define a custom tokenizer function
def custom_tokenizer(text):
    tokens = nltk.word_tokenize(text)  # Tokenize the text
    tagged_tokens = nltk.pos_tag(tokens)  # Tag the tokens with POS
    filtered_tokens = []
    for token, tag in tagged_tokens:
        # Ignore symbol tags and certain types
        if tag.isalpha() and tag not in ['SYM', 'RBS', 'RBR']:
            if tag in ['NNP', 'NNPS']:
                filtered_tokens.append('NN')
            else:
                filtered_tokens.append(tag)
    return filtered_tokens

# Extract features (text data) and target (discipline)
X_train_text_discipline = data['Content']  # Assuming 'Content' column contains text data
y_train_discipline = LabelEncoder().fit_transform(data['discipline'])

# Extract features (text data) and target (purpose)
X_train_text_purpose = data['Content']  # Assuming 'Content' column contains text data
y_train_purpose = LabelEncoder().fit_transform(data['Purpose'])

# Create Bag-of-Words (BoW) vectorizer with custom tokenizer for discipline
bow_vectorizer_discipline = CountVectorizer(tokenizer=custom_tokenizer)

# Fit and transform text data for discipline
X_train_bow_discipline = bow_vectorizer_discipline.fit_transform(X_train_text_discipline)

# Create and train a logistic regression model for discipline
model_discipline = LogisticRegression()
model_discipline.fit(X_train_bow_discipline, y_train_discipline)

# Create Bag-of-Words (BoW) vectorizer with custom tokenizer for purpose
bow_vectorizer_purpose = CountVectorizer(tokenizer=custom_tokenizer)

# Fit and transform text data for purpose
X_train_bow_purpose = bow_vectorizer_purpose.fit_transform(X_train_text_purpose)

# Create and train a logistic regression model for purpose
model_purpose = LogisticRegression()
model_purpose.fit(X_train_bow_purpose, y_train_purpose)

# Define Flask application
app = Flask(__name__)

# Route for starting page
@app.route('/', methods=['GET'])
def start_page():
    return render_template('start_page.html')

# Route for about page
@app.route('/about_page', methods=['GET'])
def start_page():
    return render_template('about_page.html')

# Route for discipline prediction
@app.route('/predict_discipline', methods=['GET', 'POST'])
def predict_discipline():
    if request.method == 'POST':
        input_text = request.form.get('text')
        if input_text:
            try:
                # Transform text data using BoW vectorizer for discipline
                input_bow_discipline = bow_vectorizer_discipline.transform([input_text])

                # Predict the discipline based on the uploaded text data
                prediction_probabilities_discipline = model_discipline.predict_proba(input_bow_discipline)[0]
                predicted_discipline = model_discipline.classes_[prediction_probabilities_discipline.argmax()]

                # Calculate percentages
                percentages_discipline = {label: probability * 100 for label, probability in zip(model_discipline.classes_, prediction_probabilities_discipline)}

                # Get top 5 POS tags and their examples
                top_tags_discipline, tag_examples_discipline = get_top_pos_tags(input_text)

                return render_template('index_nltk_discipline.html', predictions=[{
                    'file_name': 'User Input',
                    'discipline': predicted_discipline,
                    'probability': prediction_probabilities_discipline.max(),
                    'percentages': percentages_discipline,
                    'top_tags': top_tags_discipline,
                    'tag_examples': tag_examples_discipline
                }])
            except Exception as e:
                return render_template('index_nltk_discipline.html', error_message=str(e))
        else:
            return render_template('index_nltk_discipline.html', error_message='No text data received')
    
    return render_template('index_nltk_discipline.html', predictions=None, error_message=None)

# Route for purpose prediction
@app.route('/predict_purpose', methods=['GET', 'POST'])
def predict_purpose():
    if request.method == 'POST':
        input_text = request.form.get('text')
        if input_text:
            try:
                # Transform text data using BoW vectorizer for purpose
                input_bow_purpose = bow_vectorizer_purpose.transform([input_text])

                # Predict the purpose based on the uploaded text data
                prediction_probabilities_purpose = model_purpose.predict_proba(input_bow_purpose)[0]
                predicted_purpose = model_purpose.classes_[prediction_probabilities_purpose.argmax()]

                # Calculate percentages
                percentages_purpose = {label: probability * 100 for label, probability in zip(model_purpose.classes_, prediction_probabilities_purpose)}

                # Get top 5 POS tags and their examples
                top_tags_purpose, tag_examples_purpose = get_top_pos_tags(input_text)

                return render_template('index_nltk_purpose.html', predictions=[{
                    'file_name': 'User Input',
                    'purpose': predicted_purpose,
                    'probability': prediction_probabilities_purpose.max(),
                    'percentages': percentages_purpose,
                    'top_tags': top_tags_purpose,
                    'tag_examples': tag_examples_purpose
                }])
            except Exception as e:
                return render_template('index_nltk_purpose.html', error_message=str(e))
        else:
            return render_template('index_nltk_purpose.html', error_message='No text data received')
    
    return render_template('index_nltk_purpose.html', predictions=None, error_message=None)

def get_top_pos_tags(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Tag the tokens with POS
    tagged_tokens = nltk.pos_tag(tokens)
    # Filter out unwanted tags
    filtered_tokens = [(token.lower(), tag) for token, tag in tagged_tokens if tag.isalpha() and tag not in ['SYM', 'RBS', 'RBR']]
    # Count the occurrences of each tag
    tag_counts = {}
    for _, tag in filtered_tokens:
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    # Sort the tags by frequency
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    # Extract top 5 tags
    top_tags = [tag for tag, _ in sorted_tags[:5]]
    # Find examples for each top tag
    tag_examples = {}
    for tag in top_tags:
        examples = set([token for token, t in filtered_tokens if t == tag])
        tag_examples[tag] = list(examples)[:5]  # Get the first 5 unique examples
    return top_tags, tag_examples

if __name__ == '__main__':
    app.run(host='localhost', port=5023)
