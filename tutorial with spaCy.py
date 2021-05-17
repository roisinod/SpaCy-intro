import spacy
from spacy.matcher import PhraseMatcher
nlp = spacy.load('en_core_web_sm')
doc = nlp("Tea is healthy and calmed, don't you think?")
for token in doc:
    print(token)
#does not print in list
#lemmatizing.  The "lemma" of a word is its base form. For example, "walk" is the lemma of the word "walking". So, when you lemmatize the word walking, you would convert it to walk.
print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
#does not see punctuation as stopword
print("-"*40)
for token in doc:
    print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")
#pattern matching - PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
patterns = [nlp(text) for text in terms]
matcher.add("TerminologyList", patterns)
text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
               "photography tests pitting the iPhone 11 Pro against the "
               "Galaxy Note 10 Plus and last year’s iPhone XS and Google Pixel 3.")
matches = matcher(text_doc)
print(matches)
match_id, start, end = matches[0]
print(nlp.vocab.strings[match_id], text_doc[start:end])
