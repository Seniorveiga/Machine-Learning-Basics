

#---object_to_work_with--------

document = ["cats say meow","dogs say woof", "dogs chasing cats"]

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer()

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(document)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names_out()

# Print words
print(words)

"""
NOTES: One of the functions, "get_feautre_names()" has been updated to "get_feature_names_out()".

The matrix informs us on how the words are distributed.
We can know how many words a text have by counting the elements of the matrix. Each element that is not 0 is an element 
that appears on the text.
"""