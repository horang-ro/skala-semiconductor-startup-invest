"""Official BGE-M3 dense+sparse inference; normalized float32 + portable SQLite."""
import argparse
import hashlib
import importlib.metadata
import json
import sqlite3
import time
from pathlib import Path
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

ROOT=Path(__file__).resolve().parent
MODEL_ID='BAAI/bge-m3'
MODEL_REVISION='5617a9f61b028005a4858fdac845db406aefb181'

def load_model(path=None,threads=4):
    torch.set_num_threads(threads)
    if path is None:
        from huggingface_hub import snapshot_download
        path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,
            allow_patterns=['*.json','pytorch_model.bin','sparse_linear.pt','colbert_linear.pt','sentencepiece.bpe.model','1_Pooling/*'])
    return BGEM3FlagModel(str(path),use_fp16=False,devices=['cpu'],normalize_embeddings=True)

def encode(model,texts,batch_size=8,max_length=1024):
    sizes=[len(t) for t in model.tokenizer(texts,truncation=False)['input_ids']]
    if max(sizes)>max_length:
        raise ValueError(f'Would truncate {max(sizes)} tokens; raise max_length or split facts.')
    output=model.encode(texts,batch_size=batch_size,max_length=max_length,
                        return_dense=True,return_sparse=True,return_colbert_vecs=False)
    dense=np.asarray(output['dense_vecs'],dtype=np.float32)
    sparse=[{str(k):float(v) for k,v in w.items()} for w in output['lexical_weights']]
    return dense,sparse,sizes

def run(data,model_path,batch_size,threads):
    chunks=[json.loads(l) for l in (data/'chunks.jsonl').read_text().splitlines()]
    model=load_model(model_path,threads);texts=[c['embedding_text'] for c in chunks]
    token_lengths=[len(x) for x in model.tokenizer(texts,truncation=False)['input_ids']]
    print(f'MODEL_READY chunks={len(chunks)} max_tokens={max(token_lengths)}',flush=True)
    limit=max(512,max(token_lengths))
    if limit>8192: raise ValueError('Fact exceeds model capacity; split it before embedding')
    start=time.monotonic();dense,sparse,_=encode(model,texts,batch_size,limit)
    assert dense.shape==(len(chunks),1024) and np.isfinite(dense).all()
    norms=np.linalg.norm(dense,axis=1)
    assert np.allclose(norms,1,atol=1e-5) and all(sparse)
    np.save(data/'dense_vectors.npy',dense,allow_pickle=False)
    ids=[c['chunk_id'] for c in chunks]
    (data/'vector_ids.json').write_text(json.dumps(ids,ensure_ascii=False,indent=2)+'\n')
    with (data/'sparse_vectors.jsonl').open('w') as sf,(data/'embeddings.jsonl').open('w') as ef:
        for i,c in enumerate(chunks):
            sf.write(json.dumps({'chunk_id':ids[i],'weights':sparse[i]},ensure_ascii=False)+'\n')
            # Named vector keys work as a portable vector DB import payload.
            ef.write(json.dumps({'id':ids[i],'vector':{'dense':dense[i].tolist(),
                  'sparse':{'indices':[int(k) for k in sparse[i]],'values':list(sparse[i].values())}},
                  'payload':{'text':c['embedding_text'],**c['metadata']}},ensure_ascii=False)+'\n')
    db=data/'vector_store.sqlite'
    if db.exists():db.unlink()
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE chunks (chunk_id TEXT PRIMARY KEY, company_id TEXT, segment TEXT, category TEXT, text TEXT, metadata_json TEXT, dense_vector BLOB, sparse_json TEXT)')
        conn.executemany('INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)',[
            (c['chunk_id'],c['metadata']['company_id'],c['metadata']['segment'],c['metadata']['category'],c['embedding_text'],
             json.dumps(c['metadata'],ensure_ascii=False),dense[i].tobytes(),json.dumps(sparse[i])) for i,c in enumerate(chunks)])
        conn.execute('CREATE INDEX company_filter ON chunks(company_id)')
        conn.execute('CREATE INDEX category_filter ON chunks(category)')
        conn.execute('CREATE INDEX segment_filter ON chunks(segment)')
    manifest={'status':'completed','model_id':MODEL_ID,'model_revision':MODEL_REVISION,
      'embedding_library':'FlagEmbedding','dimension':1024,'dense_dtype':'float32',
      'dense_normalization':'L2','dense_similarity':'cosine / normalized inner product',
      'sparse_representation':'BGE-M3 learned token-id lexical weights','colbert_generated':False,
      'chunk_count':len(chunks),'max_input_tokens':max(token_lengths),'encoding_max_length':limit,'truncated_chunks':0,
      'batch_size':batch_size,'device':'cpu','elapsed_seconds':round(time.monotonic()-start,2),
      'source_document_as_of':'2026-09-29','processed_at':'2026-09-30','norm_min':float(norms.min()),'norm_max':float(norms.max()),
      'sparse_nonzero_total':sum(len(w) for w in sparse),'chunk_order':'vector_ids.json',
      'input_sha256':hashlib.sha256((data/'chunks.jsonl').read_bytes()).hexdigest(),
      'packages':{p:importlib.metadata.version(p) for p in ['FlagEmbedding','torch','transformers','numpy','huggingface-hub']}}
    (data/'embedding_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    from build_faiss import build
    build(data)
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--model-path',type=Path)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();run(a.data,a.model_path,a.batch_size,a.threads)
