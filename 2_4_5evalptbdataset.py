import sys
sys.path.append('..')
import numpy as np
from common.utils import preprocess, create_co_matrix, most_similar, ppmi
from dataset import ptb
window_size = 2
wordvec_size = 100
# Load PTB dataset
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print("Counting co-occurrence..")
C = create_co_matrix(corpus, vocab_size, window_size)
print("Calculating PPMI..")
W = ppmi(C, verbose=True)
print("Calculating SVD..")
try:
    #trancated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W, n_components=wordvec_size)
except ImportError:
    #fallback to slow SVD
    print("Warning: sklearn not found. Using slow SVD.")
    U, S, V = np.linalg.svd(W)    
print("Calculating word vectors..")
word_vecs = U[:, :wordvec_size]
querys = ['you','year','car','toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs,top=5)

