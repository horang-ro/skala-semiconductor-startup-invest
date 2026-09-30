"""CPU local hybrid retrieval with exact metadata filters and RRF fusion."""
import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
import faiss
from embed import load_model,encode

ROOT=Path(__file__).resolve().parent

def search(db,model,query,top_k=5,company=None,segment=None,category=None,feature=None,agent=None):
    dense,sparse,_=encode(model,[query],batch_size=1,max_length=8192)
    return search_vectors(db,dense[0],sparse[0],top_k,company,segment,category,feature,agent)

def search_vectors(db,query_dense,query_sparse,top_k=5,company=None,segment=None,category=None,feature=None,agent=None):
    clauses=[];args=[]
    if segment:clauses.append('segment=?');args.append(segment)
    if category:clauses.append('category=?');args.append(category)
    sql='SELECT chunk_id,company_id,text,metadata_json,dense_vector,sparse_json FROM chunks'
    if clauses:sql+=' WHERE '+' AND '.join(clauses)
    with sqlite3.connect(db) as conn:rows=conn.execute(sql,args).fetchall()
    selected=[]
    for row in rows:
        m=json.loads(row[3])
        if company and company not in m['company_ids'] and company!=m['company_id']:continue
        if feature and feature not in m['features']:continue
        if agent and agent not in m['agent_routes']:continue
        selected.append((row,m))
    if not selected:return []
    # Search all flat-index vectors, then restrict dense ranking to exact filters.
    # No approximate top-k prefilter is used, so no matching company is lost.
    ids=json.loads((Path(db).parent/'vector_ids.json').read_text())
    index=faiss.read_index(str(Path(db).parent/'dense.faiss'))
    distances,positions=index.search(np.ascontiguousarray(query_dense[None,:],dtype=np.float32),index.ntotal)
    byid={ids[int(pos)]:float(distance) for pos,distance in zip(positions[0],distances[0])}
    ds=np.array([byid[row[0]] for row,_ in selected])
    ss=np.array([sum(v*query_sparse.get(k,0) for k,v in json.loads(r[0][5]).items()) for r in selected])
    # Raw dense and sparse scores have different scales; combine rankings instead.
    scores=np.zeros(len(selected))
    for order in [np.argsort(-ds,kind='stable'),np.argsort(-ss,kind='stable')]:
        for rank,idx in enumerate(order,1):scores[idx]+=1/(60+rank)
    result=[]
    for idx in np.argsort(-scores,kind='stable')[:top_k]:
        row,m=selected[idx]
        result.append({'chunk_id':row[0],'company_id':row[1],'text':row[2],
                       'rrf_score':float(scores[idx]),'dense_score':float(ds[idx]),'sparse_score':float(ss[idx]),
                       'metadata':m})
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('query');p.add_argument('--db',type=Path,default=ROOT/'data/vector_store.sqlite')
    p.add_argument('--model-path',type=Path);p.add_argument('--company',help='Company code, e.g. DX, HA, GRQ')
    p.add_argument('--segment',choices=list('ABCD'));p.add_argument('--category');p.add_argument('--feature');p.add_argument('--agent')
    p.add_argument('--top-k',type=int,default=5);a=p.parse_args()
    print(json.dumps(search(a.db,load_model(a.model_path),a.query,a.top_k,a.company,a.segment,a.category,a.feature,a.agent),ensure_ascii=False,indent=2))
