import numpy as np
def preprocess(text):
    text = text.lower()
    text = text.replace('.',' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus,word_to_id,id_to_word

def create_co_matrix(corpus,vocab_size,window_size=1):
    if window_size < 1:
        raise ValueError("window_size must be a positive integer")
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size,vocab_size),dtype=np.int32)
    
    for idx,word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            right_idx = idx + i 
            left_idx = idx - i           
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id,left_word_id] += 1
        
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id,right_word_id] += 1       
    return co_matrix

def cos_similarity(x, y, eps=1e-8):
    nx = x/np.sqrt(np.sum(x**2) + eps)
    ny = y/np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx,ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''
    Find most similar words
    query: word to find similar words
    word_to_id: dictionary of word to id
    id_to_word: dictionary of id to word
    word_matrix: word vector matrix
    top: number of similar words to find
    '''

    if query not in word_to_id:
        print('%s is not found' % query)
        return
    print('\n[query] '+query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    #cos similarity
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # result of the calc of cos similarity, return the words from top 
    count = 0
    for i in (-1*similarity).argsort():
        if id_to_word[i]==query:
            continue #queryについてはスキップする
        print(' %s: %s' % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            break
    return

def ppmi(C,verbose=False,eps=1e-8):
    return 0
