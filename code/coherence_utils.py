from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

def compute_coherence_for_one(dictionary, corpus, texts, num_topics):
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=3)
    coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence = coherencemodel.get_coherence()
    return (model, coherence, num_topics)