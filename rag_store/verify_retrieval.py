"""Targeted retrieval smoke checks, not a production accuracy benchmark."""
import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np
import faiss
from embed import load_model,encode
from search import search_vectors

ROOT=Path(__file__).resolve().parent
CASES=[
 ('DX','technology','딥엑스 DX-M1 연산 성능과 정밀도는 무엇인가요?','25 TOPS'),
 ('HA','financial','하이퍼엑셀 2025년 실제 매출은 얼마인가요?','2,244,031,820'),
 ('FUR','financial','퓨리오사AI 2025년 당기순손실과 영업손실은 얼마인가요?','8,333.5'),
 ('GRQ','financial','Groq 2025년 5억 달러는 실제 매출인가요 전망치인가요?','전망치'),
 ('RAIN',None,'Rain AI 특허 소유권과 OpenAI 양도 리스크는 무엇인가요?','양도'),
 ('QD',None,'Quadric은 칩 판매 기업인가요 IP 라이선스 기업인가요?','라이선스'),
 ('DM','technology','d-Matrix의 메모리는 온칩 SRAM인가요 HBM인가요?','SRAM'),
 ('PAN','funding','파네시아 R&D 과제 30M 달러는 투자금인가요 지원금인가요?','과제'),
]

def run(data,model_path):
    model=load_model(model_path)
    dense,sparse,_=encode(model,[c[2] for c in CASES],batch_size=8,max_length=512)
    tests=[]
    for i,(company,category,q,needle) in enumerate(CASES):
        hits=search_vectors(data/'vector_store.sqlite',dense[i],sparse[i],5,company=company,category=category)
        assert hits and all(h['metadata']['company_id']==company or company in h['metadata']['company_ids'] for h in hits)
        found=any(needle in h['text'] for h in hits)
        tests.append({'query':q,'company_filter':company,'category_filter':category,
            'expected_fragment':needle,'expected_fragment_found_at_5':found,
            'hits':[{'chunk_id':h['chunk_id'],'text':h['text'],'rrf_score':h['rrf_score'],
                     'source_ids':h['metadata']['citation_ids']} for h in hits]})
    vectors=np.load(data/'dense_vectors.npy',allow_pickle=False)
    ids=json.loads((data/'vector_ids.json').read_text())
    with sqlite3.connect(data/'vector_store.sqlite') as conn:
        count=conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
        blob=conn.execute('SELECT dense_vector FROM chunks WHERE chunk_id=?',(ids[0],)).fetchone()[0]
    assert count==len(ids)==len(vectors) and np.array_equal(np.frombuffer(blob,dtype=np.float32),vectors[0])
    index=faiss.read_index(str(data/'dense.faiss'))
    assert index.ntotal==count and index.d==1024
    distances,positions=index.search(vectors[:3],1)
    assert np.array_equal(positions[:,0],np.arange(3)) and np.allclose(distances[:,0],1,atol=1e-5)
    rev=json.loads((data/'revenue_records.json').read_text())
    assert next(r for r in rev if r['company_id']=='GRQ' and r['revenue_year']==2025)['metric_type']=='forecast'
    assert next(r for r in rev if r['company_id']=='HA' and r['revenue_year']==2025)['amount']==2244031820
    assert next(r for r in rev if r['company_id']=='IHW' and r['revenue_year']==2025)['amount'] is None
    assert len([r for r in rev if r['company_id']=='DX' and r['revenue_year']==2024])==2
    output={'kind':'retrieval_smoke_test_not_accuracy_benchmark','test_count':len(tests),
      'passed':sum(t['expected_fragment_found_at_5'] for t in tests),'tests':tests,
      'vector_database_alignment':'passed','financial_semantics_checks':'passed'}
    (data/'retrieval_checks.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k!='tests'},ensure_ascii=False,indent=2))
    assert output['passed']==len(tests),'Inspect retrieval_checks.json'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data');p.add_argument('--model-path',type=Path)
    a=p.parse_args();run(a.data,a.model_path)
