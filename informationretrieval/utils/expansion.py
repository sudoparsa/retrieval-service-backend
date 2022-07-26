import numpy as np


# Query expansion for fasttext and transformers
def Rocchio(model, query, k=10):
    indices = model.most_similar(query, False, 2 * k)[0]
    relevant_ind = indices[:k]
    irrelevant = indices[k:]
    embedding_rel = [model.doc_embedding['embedding'][i] for i in relevant_ind]
    embedding_irrel = [model.doc_embedding['embedding'][i] for i in irrelevant]
    rel_score = np.mean(embedding_rel, axis=0)
    irrel_score = np.mean(embedding_irrel, axis=0)
    modified_query = model.embed(model.preprocessor.run(query)).reshape(1, -1) + rel_score - irrel_score
    return modified_query
