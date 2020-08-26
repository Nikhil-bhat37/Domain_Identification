import tkinter as tk
from tkinter import *
doc=tk.Tk()
def domain():
    import glob
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from sklearn.feature_extraction.text import CountVectorizer
    from IPython.core.interactiveshell import InteractiveShell
    import numpy as np
    import string
    InteractiveShell.ast_node_interactivity = "all"
    doc=[]
    from pathlib import Path
    mypath = Path().absolute()
    test=e1.get()
    test1=mypath/'Desktop'/'Test'/test
    with open(test1) as f:
            raw = open(test1, 'rU').read().splitlines()
            mystr ='.'.join([line.strip() for line in raw])
            doc.append(mystr)
        
    path=r'p2\*.txt'
    files=glob.glob(path)
    for name in files:
        with open(name) as f:
            raw = open(name, 'rU').read().splitlines()
            mystr ='.'.join([line.strip() for line in raw])
            doc.append(mystr) 
    print(doc)
    lemmer = nltk.stem.WordNetLemmatizer()
    def LemTokens(tokens):
         return [lemmer.lemmatize(token) for token in tokens]
    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
    def LemNormalize(text):
         return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    from sklearn.feature_extraction.text import CountVectorizer
    LemVectorizer = CountVectorizer(tokenizer=LemNormalize, stop_words='english')
    LemVectorizer.fit_transform(doc)
    print(LemVectorizer.vocabulary_)
    tf_matrix = LemVectorizer.transform(doc).toarray()
    print(tf_matrix)
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    print(tfidf_matrix)
    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    print ((cos_similarity_matrix))
    sim_doc=cos_similarity_matrix[0:1]
    print(sim_doc)
    a=sim_doc.tolist()
    b=np.concatenate([np.array(i) for i in sim_doc])
    c=b.tolist()
    c.pop(0)
    d=c.index(max(c))
    e=max(c)
    if(e==0):
            messagebox.showinfo("Domain","Empty document")
    else:
        if(d<=5):
             messagebox.showinfo("Domain", "POLITICS")  
        else:
            messagebox.showinfo("Domain", "SPORTS")

doc.title('Domain Identification') 
Label(doc, text='File Name',height=3).grid(row=0) 
e1 = Entry(doc)  
e1.grid(row=0, column=1) 
Button(doc, text='Submit', command=domain).grid(row=4, column=1, sticky=W, pady=4)

doc.mainloop()