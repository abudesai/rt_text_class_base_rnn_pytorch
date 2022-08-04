
import numpy as np, pandas as pd
import nltk
import re
import sys , os
import operator
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

stop_words_path = os.path.join(os.path.dirname(__file__), 'stop_words.txt')
stopwords = set(w.rstrip() for w in open(stop_words_path))
porter_stemmer=nltk.PorterStemmer()


PADDING_IDX = 0
UNKNOWN_IDX = 1




def tokenize(document):
    tokens = re.sub(r"[^A-Za-z0-9\-]", " ", document).lower().split()
    tokens = [t for t in tokens if len(t) > 2] # remove short words, they're probably not useful
    # tokens = [t for t in tokens if t not in stopwords] # remove stopwords  !! seems to do better if we keep stop words !!
    porter_stemmer=nltk.PorterStemmer()
    tokens = [porter_stemmer.stem(t) for t in tokens]
    return tokens


class CustomTokenizerWithLimitedVocab(BaseEstimator, TransformerMixin):
    '''
    Tokenizes the text column in given dataframe. 
    Can be used to limit the vocabulary size. 
    text_col: name of field with text in dataframe
    vocab_size: max vocab size to use
    keep_words: words to keep regardless of their frequency
    start_token: token used to indicate start of text (if any)
    end_token: token used to indicate end of text (if any)    
    
    Original code referenced from here: 
    https://github.com/lazyprogrammer/machine_learning_examples/blob/master/rnn_class/brown.py
    
    '''   
    
    def __init__(self, text_col, vocab_size=5000, keep_words=[]):
        self.text_col = text_col
        self.vocab_size = vocab_size
        self.keep_words = keep_words
        
        
    def fit(self, data):
        sentences = list(data[self.text_col])
        
        word2idx = {}
        word_idx_count = {}
        idx2word = []
        i = 0  
  
        for sentence in sentences:
            tokens = tokenize(sentence)
            for token in tokens:
                if token not in word2idx:
                    idx2word.append(token)
                    word2idx[token] = i
                    i += 1 
                
                # keep track of counts for later sorting
                idx = word2idx[token]
                word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
        
        # set all the words we want to keep to infinity
        # so that they are included when we pick the most
        # common words
        for word in self.keep_words:
            word_idx_count[word2idx[word]] = float('inf')
        
        # sort words in decreasing order of counts
        sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
        
        word2idx_small = {}
        new_idx = 2     # reserve 0 and 1 for padding and unknown
        idx_new_idx_map = {}
        for idx, count in sorted_word_idx_count[:self.vocab_size-2]:
            word = idx2word[idx]
            word2idx_small[word] = new_idx
            idx_new_idx_map[idx] = new_idx
            new_idx += 1
            
        
        # let 'unknown' be 1, and padding be 0
        word2idx_small['UNKNOWN'] = UNKNOWN_IDX 
        self.unknown = UNKNOWN_IDX        
        self.word2idx_small = word2idx_small
        
        return self
    
        
    def transform(self, data): 
        sentences = list(data[self.text_col])
        sentences_small = []
        for sentence in sentences:
            tokens = tokenize(sentence)
            new_sentence = [ self.word2idx_small[token]
                                if token in self.word2idx_small else self.unknown
                                for token in tokens]
            sentences_small.append(new_sentence)
        
        data[self.text_col] = sentences_small   
        return data



def pad_or_truncate_sequence(tokens, max_len): 
    if len(tokens) < max_len: 
        tokens = [ PADDING_IDX ] * (max_len - len(tokens)) + tokens
        return tokens
    elif len(tokens) > max_len: 
        return tokens[- max_len : ]
    return tokens


class PadderTruncator(BaseEstimator, TransformerMixin): 
    def __init__(self, text_col, max_len) -> None:
        super().__init__()
        self.text_col = text_col
        self.max_len = max_len
    
    def fit(self, data): return self
    
    def transform(self, data): 
        num_tokens = data[self.text_col].apply(
            lambda tokens: len(tokens)
        )
        max_tokens = min(num_tokens.max() , self.max_len)
        
        data[self.text_col] = data[self.text_col].apply(
            lambda tokens: pad_or_truncate_sequence(tokens, max_tokens)
        )       
        return data


class TargetFeatureAdder(BaseEstimator, TransformerMixin): 
    def __init__(self, target_col, fill_value) -> None:
        super().__init__()
        self.target_col = target_col
        self.fill_value = fill_value
    
    def fit(self, data): return self
    
    def transform(self, data): 
        if self.target_col not in data.columns: 
            data[self.target_col] = self.fill_value
        return data



class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, col):
        self.col = col
        
    def fit(self, X, y=None):  return self
    
    def transform(self, X): return X[self.col]
    
    

class ColumnsSelector(BaseEstimator, TransformerMixin):
    """Select only specified columns."""
    def __init__(self, columns, selector_type='keep'):
        self.columns = columns
        self.selector_type = selector_type
        
        
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):  
        if self.selector_type == 'keep':
            retained_cols = [col for col in X.columns if col in self.columns]
            X = X[retained_cols].copy()
        elif self.selector_type == 'drop':
            dropped_cols = [col for col in X.columns if col in self.columns]  
            X = X.drop(dropped_cols, axis=1)      
        else: 
            raise Exception(f'''
                Error: Invalid selector_type. 
                Allowed values ['keep', 'drop']
                Given type = {self.selector_type} ''')   
        return X
    
 
class CustomLabelEncoder(BaseEstimator, TransformerMixin): 
    def __init__(self, target_col, dummy_label) -> None:
        super().__init__()
        self.target_col = target_col
        self.dummy_label = dummy_label
        self.lb = LabelEncoder()


    def fit(self, data):                
        self.lb.fit(data[self.target_col])             
        self.classes_ = self.lb.classes_ 
        return self 
    
    
    def transform(self, data): 
        check_val_if_pred = data[self.target_col].values[0]
        if self.target_col in data.columns and check_val_if_pred != self.dummy_label: 
            data[self.target_col] = self.lb.transform(data[self.target_col])
        return data



class XYSplitter(BaseEstimator, TransformerMixin): 
    def __init__(self, text_col, target_col, id_col):
        self.text_col = text_col
        self.target_col = target_col
        self.id_col = id_col
    
    def fit(self, data): return self
    
    def transform(self, data): 
        if self.target_col in data.columns: 
            y = data[self.target_col].values
        else: 
            y = None
        
        X = np.array(data[self.text_col].values.tolist())        
        return { 'X': X, 'y': y  }
    
        
    