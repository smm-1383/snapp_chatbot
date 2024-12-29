from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
with open('..\\docs.json', 'r+') as fo:
    docs = json.load(fo)

k1 = docs.keys()
full_list = []
for i in k1:
    k2 = docs[i].keys()
    for j in k2:
        full_list.append((i, j, docs[i][j]))
    # break

embeddings = {}
full_list_doc = {}
for t, st, doc in full_list:
    tdoc = 'عنوان گروه صفحه: ' + t + 'عنوان این صفحه: ' + st + '\n\n' + doc
    e = model.encode(tdoc)
    embeddings[t + ' ' + st] = e.astype(float).tolist()
    full_list_doc[t + ' ' + st] = tdoc
    # break

with open('embeddings.json', 'w', encoding='utf-8') as f:
    json.dump(embeddings, f)

with open('full_doc.json', 'w', encoding='utf-8') as f:
    json.dump(full_list_doc, f)