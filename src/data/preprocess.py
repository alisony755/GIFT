import re
import nltk


class Preprocessor:
    def __init__(self):
        pass

    # Clean raw text
    def clean(self, text):
        text = text.lower() # Convert to lowercase
        text = re.sub(r"http\S+", "", text) # Remove urls
        text = re.sub(r"[^a-z0-9' ]", " ", text) # Keep letters, numbers, apostrophes
        text = re.sub(r"\s+", " ", text) # Remove/standardize whitespace

        return text.strip()

    # Tokenize corpus
    def tokenize(self, texts):
        tokenized = []

        for text in texts:
            clean_text = self.clean(text)
            tokens = nltk.word_tokenize(clean_text)
            tokenized.append(tokens)

        return tokenized

    # POS tagging
    def pos_tag(self, tokenized_texts):
        corpus_tags = []

        for i, tokens in enumerate(tokenized_texts):

            if i % 50 == 0:
                print(f"[POS] processing doc {i}/{len(tokenized_texts)}")

            # Safety check
            if len(tokens) == 0:
                corpus_tags.append([])
                continue

            tags = nltk.pos_tag(tokens)

            only_tags = [
                tag
                for _, tag in tags
            ]

            corpus_tags.append(only_tags)

        # Check if each doc has 1 POS list
        assert len(corpus_tags) == len(tokenized_texts)

        return corpus_tags