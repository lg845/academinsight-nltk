import nltk

def custom_tokenizer(text):
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    filtered_tokens = []
    for token, tag in tagged_tokens:
        if tag.isalpha() and tag not in ['SYM', 'RBS', 'RBR']:
            if tag in ['NNP', 'NNPS']:
                filtered_tokens.append('NN')
            else:
                filtered_tokens.append(tag)
    return filtered_tokens
