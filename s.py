import numpy as np
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
from common.utils import preprocess, create_co_matrix, most_similar, ppmi

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
#print(corpus,word_to_id,id_to_word)
C = create_co_matrix(corpus,vocab_size)
#most_similar('you',word_to_id, id_to_word, C,top=5)
W = ppmi(C)
np.set_printoptions(precision=3)
#print(C)
#print('-'*50)
#print('PPMI')

#SVD
U,S,V = np.linalg.svd(W)

print("共起行列: ", C[0])
print("PPMI行列: ", W[0])
print("SVD: ", U[0])
print(U[0, :2])

for word,word_id in word_to_id.items():
    plt.annotate(word,(U[word_id,0],U[word_id,1]))
plt.scatter(U[:,0],U[:,1],alpha=0.5)
plt.show()



