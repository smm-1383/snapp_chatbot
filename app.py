from langchain.schema.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

import json

oak = '<OPENAI_API_KEY>'
base = 'https://api.avalai.ir/v1'
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

with open('embeddings.json', 'r+') as fo:
    embeddings = json.load(fo)

with open('full_doc.json', 'r+') as fo:
    data = json.load(fo)


query = 'ساعت کاری منعطف در اسنپ چطور است؟'
while query:
    query_embedding = model.encode(query)

    similarities = {
        key: cosine_similarity([query_embedding], [value])[0][0]
        for key, value in embeddings.items()
    }

    best_match_id = max(similarities, key=similarities.get)
    print('***', best_match_id)
    rag = data[best_match_id]
    # print(rag)
    sysprompt = f"""
you are a persian asssistant and you will always talk in persian.
Answer the following question based only on the provided context. 
if the question is not related to this document or to snapp in total, say you can not answer it because it's not related to snapp and do not talk about the context which is given to you. just say your question is not related to snapp
I will tip you $1000 if the user finds the answer helpful.
<context>
{rag}
</context>
search the question in the context word by word before saying it's not related.
if the question seems to be related to snapp but the answer is not in the text, ask user to write question in a different way.
"""

    prompt = ChatPromptTemplate.from_messages([
        ('system', sysprompt),
        ('user', 'Question:{question}')
    ])
    llm = ChatOpenAI(api_key=oak, base_url=base)
    out = StrOutputParser()
    chain = prompt | llm | out

    print(chain.invoke({'question': query}))
    query = input('q: ')

