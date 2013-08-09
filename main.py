import xml.etree.ElementTree as ET
import nltk
import cv2
import numpy as np
from pymorphy import get_morph
morph = get_morph('K:\databases\BAR\pymorfy')

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM2(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self):
        self.model = cv2.SVM()

    def train(self, samples, responses):
        #setting algorithm parameters
        params = dict( kernel_type = cv2.SVM_LINEAR,
                       svm_type = cv2.SVM_C_SVC,
                       C = 1 )
        self.model.train(samples, responses, params = params)

    def predict(self, samples):
        return np.float32( [self.model.predict(s) for s in samples])


tree = ET.parse('imhonet-books-short.xml')

root = tree.getroot()
books = []
book_morph = []
unique_vector = []
unique_vector_structure = []
positive_reviews = []
negative_reviews = []
i = 0 # percent

for row in root.iter('row'):
    book = []
    for child in row:
        book.append(child.text)
    books.append( book )

# sentence = books[10][4]
# sentence = sentence.replace('\\n', ' ')
# tokens = nltk.word_tokenize(sentence)
# oneWord = morph.get_graminfo(tokens[1].upper())[0];

# print(oneWord['norm'])
# print(oneWord['class'])
# print(oneWord['info'])
# print(oneWord['method'])
# print( len(books) )

#creating all features vector
for index, book in enumerate(books):
    # print(book[0])
    sentence = book[4]
    if not sentence:
        print( 'not found: ', index )
        continue
    sentence = sentence.replace('\\n', ' ')
    tokens = nltk.word_tokenize(sentence)
    for token in tokens:
        token_arr = []
        #delete too short items
        if len(token) < 4:
            continue
        vec_el = morph.get_graminfo(token.upper());
        if not vec_el:
            continue
        else:
            vec_norm = vec_el[0]['norm']
            #adding it to books
            token_arr.append(vec_norm)
        if vec_el not in unique_vector:
            unique_vector.append(vec_norm)
            unique_vector_structure.append(vec_el[0])
    books[index].append( token_arr )

#creating train and test vector

print(unique_vector[201])
print( len(unique_vector) )
#let's write a file
f = open('unique_vector.txt', 'w')
# pickle.dump(unique_vector, f)
for item in unique_vector:
  f.write("%s\n" % item.encode('utf8'))



reviews_all = []
reviews_cat = []

#second bypass
for book in books:
    sentence = book[4]
    if not sentence:
        continue
    if int(book[0]) > 8:
        reviews_cat.append(1)
    else:
        reviews_cat.append(0)

    #tokens = nltk.word_tokenize(sentence)
    tokens = book[5]

    #creating feature vector
    reviews_all.append( [0.0 for x in range( len(unique_vector) )] )
    #filling with features
    for token in tokens:
        #delete too short items
        # if len(token) < 4:
        #     continue
        # vec_el = morph.get_graminfo(token.upper());
        # if not vec_el:
        #     continue
        # else:
        #     vec_el = vec_el[0]['norm']

        # if vec_el in unique_vector:
        #     ind = unique_vector.index(vec_el)
        #     reviews_all[ len(reviews_all)-1 ][ind] = 1.0
        if token in unique_vector:
            ind = unique_vector.index(vec_el)
            reviews_all[ len(reviews_all)-1 ][ind] = 1.0

print(reviews_cat)
# print(reviews_all[3])
#Machine Learning
# classifier = cv2.NormalBayesClassifier()
# classifier.train( np.asarray(reviews_all, dtype=np.float32), np.asarray(reviews_cat, dtype=np.float32))
reviews_cat_test = reviews_cat[:100]
reviews_cat_train = reviews_cat[100:]

reviews_all_test = reviews_all[:100]
reviews_all_train = reviews_all[100:]

classifier = SVM2()
classifier.train( np.asarray(reviews_all_train, dtype=np.float32), np.asarray(reviews_cat_train, dtype=np.float32))

#predicted = classifier.predict_all( np.asarray(reviews_all_test, dtype=np.float32) )
predicted = classifier.predict( np.asarray(reviews_all_test, dtype=np.float32) )
print(predicted)


# for index, el in enumerate(reviews_cat_test):
#     predicted = classifier.predict_all( np.asarray(reviews_all_test, dtype=np.float32) )
#     print(predicted)
#     if predicted == el:
#         i=i+1

# i = 0
for index, el in enumerate(predicted):
    if el == reviews_cat_test[index]:
        i = i+1

print(i, '%')