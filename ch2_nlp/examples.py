import spacy
from nltk.tokenize import TweetTokenizer


"""Example2-1: tokenize text."""
nlp = spacy.load('en_core_web_sm')
text = "Mary, don't slap the green witch."
print([str(token) for token in nlp(text.lower())])

"""
tweet = "Snow white and the Seven Degrees #MakeAMovieCold@midnight:-)"
tokenizer = TweetTokenizer()
print(tokenizer.tokenize(tweet.lower))
"""

"""Exampel2-2: lemma and stem."""
doc = nlp(u"he was running late.")
print("lemma-----------------")
for token in doc:
    print('{} --> {}'.format(token, token.lemma_))

"""Example2-3: POS-tagging."""
print("POS-tagging-----------")
for token in doc:
    print('{} --> {}'.format(token, token.pos_))

"""Example2-4: chunking/shallow parsing."""
print("chunking--------------")
doc = nlp(u"Mary slapped the green witch.")
for chunk in doc.noun_chunks:
    print('{} --> {}'.format(chunk, chunk.label_))
