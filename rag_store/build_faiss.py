import argparse
import json
from pathlib import Path
import faiss
import numpy as np

def build(data):
    vectors=np.load(data/'dense_vectors.npy',allow_pickle=False)
    index=faiss.IndexFlatIP(vectors.shape[1]);index.add(np.ascontiguousarray(vectors,dtype=np.float32))
    faiss.write_index(index,str(data/'dense.faiss'))
    reloaded=faiss.read_index(str(data/'dense.faiss'))
    assert reloaded.ntotal==len(vectors) and np.array_equal(reloaded.reconstruct_n(0,len(vectors)),vectors)
    manifest=json.loads((data/'embedding_manifest.json').read_text())
    manifest.update({'dense_index':'FAISS IndexFlatIP','metadata_store':'SQLite',
                     'faiss_version':faiss.__version__,'faiss_vector_count':index.ntotal,
                     'hybrid_fusion':'RRF k=60; dense cosine and learned sparse dot product'})
    (data/'embedding_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(f'FAISS index saved: {index.ntotal} x {index.d}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'data')
    build(p.parse_args().data)
