import nltk
import os
import pickle
from flask import Flask, request, render_template

# Define Flask application
app = Flask(__name__)

# Load preprocessed data
with open('preprocessed_data.pkl', 'rb') as f:
    preprocessed_data = pickle.load(f)

X_train_text_discipline = preprocessed_data['X_train_text_discipline']
y_train_discipline = preprocessed_data['y_train_discipline']
X_train_text_purpose = preprocessed_data['X_train_text_purpose']
y_train_purpose = preprocessed_data['y_train_purpose']
bow_vectorizer_discipline = preprocessed_data['bow_vectorizer_discipline']
model_discipline = preprocessed_data['model_discipline']
bow_vectorizer_purpose = preprocessed_data['bow_vectorizer_purpose']
model_purpose = preprocessed_data['model_purpose']

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

# Use dynamic port binding for Heroku
port = int(os.environ.get("PORT", 5000))

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port)
