"""Loss-preserving Markdown -> company key/value JSON and semantic fact chunks.
No LLM or external company lookup is used; evidence is as stated in the input.
"""
import argparse
import hashlib
import json
import re
from decimal import Decimal
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEGMENTS = {"A": "엣지·온디바이스·차량 NPU", "B": "데이터센터 AI 추론·학습 가속기",
            "C": "인메모리 연산(CIM/PIM)", "D": "AI 인프라(인터커넥트·메모리·DPU·GPU IP)"}
REF = re.compile(r"\[([A-Z]+-\d+)\]")

def clean(s):
    return re.sub(r"[*`]", "", s).strip()

def refs(s):
    result = REF.findall(s)
    for m in re.finditer(r"\[([A-Z]+)-(\d+)\]\s*[~～–—-]\s*\[(?:\1)-(\d+)\]", s):
        result += [f"{m[1]}-{n}" for n in range(int(m[2]), int(m[3])+1)]
    return sorted(set(result))

def markers(text):
    # These are non-exclusive literal markers, not automatic truth judgements.
    checks = {"undisclosed_or_unknown": r"미공개|비공개|확인 안 됨|확인되지|정보 없음|표시 없음|확인하지 못|원자료에 없음",
              "forecast_or_plan": r"목표|전망|예정|계획|추진|곧 출시|개발 중",
              "estimate": r"추정", "company_claim": r"회사 주장|회사 측정|회사 제시|회사 발표|회사 설명",
              "unverified": r"출처 불명|확인 필요|재확인|재조사|미열람|제목만|대조 전|확인되지",
              "unaudited": r"감사받지 않", "source_analysis": r"Claude 의견|원자료의 분석|출처 없음"}
    return [k for k,p in checks.items() if re.search(p,text)]

def category(title):
    for prefix,key in [("기술 평가", "technical_evidence"),("기술", "technology"),
                       ("개요", "overview"),("투자", "funding"),("재무", "financial"),
                       ("상용화", "commercialization"),("리스크", "risks"),
                       ("제품 공급", "commercialization"),("양도된 특허", "ip_ownership")]:
        if title.startswith(prefix): return key
    return "context"

def entities(text, tags):
    found=[]
    for cid,c in tags.items():
        for alias in [c["name"],*c["aliases"]]:
            pattern=re.escape(alias)
            if alias[0].isascii(): pattern=r"(?<![A-Za-z0-9])"+pattern+r"(?![A-Za-z0-9])"
            if re.search(pattern,text,re.I): found.append(cid); break
    return found

def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def build(source,out):
    out.mkdir(parents=True,exist_ok=True)
    raw=source.read_text(encoding="utf-8"); lines=raw.splitlines()
    tags=json.loads((ROOT/"company_tags.json").read_text())
    refs_start=next(i for i,l in enumerate(lines) if l.startswith("## 9."))
    date=re.search(r"조사 기준일\*\*: (\d{4}-\d{2}-\d{2})",raw)[1]
    provenance={"filename":source.name,"file_id":"file_0000000031ac8209b4cd1a43b1338b41",
                "sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"document_as_of":date,
                "processed_at":"2026-09-30","external_company_sources_verified":False}
    sources={}
    for i,l in enumerate(lines[refs_start:],refs_start):
        m=re.match(r"- \[([A-Z]+-\d+)\]\s+`([^`]+)`\s*(.*)",l)
        if m:
            urls=[u.rstrip(").,，") for u in re.findall(r"https?://[^\s]+",m[3])]
            dates=re.findall(r"\b\d{4}-\d{2}-\d{2}\b",m[3])
            sources[m[1]]={"grade":m[2],"description_raw":m[3],"urls":urls,
                           "source_date":dates[0] if dates else None,"line":i+1,
                           "verification":"citation_transcribed_not_reverified"}
    companies={c["name"]:{"company_id":cid,**c,"segment_label":SEGMENTS[c["segment"]],
                          "tag_basis":"source_derived_classification_not_verified_facts",
                          "eligibility_flags":c.get("eligibility_flags",[]),"facts":{},"chunk_ids":[]}
               for cid,c in tags.items()}
    byid={c["company_id"]:c for c in companies.values()}
    chunks=[]; covered=set(); h2=""; h3=""; h4=""; cid=None; stack=[]; intro=[]
    i=0
    while i<refs_start:
        l=lines[i]
        if l.startswith("## "):
            h2=l[3:];h3=h4="";cid=None;stack=[];intro=[];i+=1;continue
        if l.startswith("### "):
            h3=l[4:];h4="";stack=[];intro=[];cid=None
            if re.match(r"[2-5]\.\d+ ",h3):
                # Match longest names first to keep aliases such as Positron unambiguous.
                candidates=entities(h3,tags)
                if len(candidates)!=1: raise ValueError((i+1,h3,candidates))
                cid=candidates[0]
            i+=1;continue
        if l.startswith("#### "):
            h4=l[5:];stack=[];intro=[];i+=1;continue
        if not l.strip() or l.strip()=="---": i+=1;continue
        start=i; kind="statement"; kv=None; parents=[]
        if l.startswith("```"):
            i+=1
            while i<refs_start and not lines[i].startswith("```"): i+=1
            i+=1;kind="template"
            text="\n".join(lines[start:i]);stack=[]
        elif l.startswith("|") and i+1<refs_start and re.match(r"\|[-:| ]+\|",lines[i+1]):
            # Header and separator are context; each data row is an independent unit.
            headers=[clean(x) for x in l.strip("|").split("|")];i+=2
            while i<refs_start and lines[i].startswith("|"):
                cells=[x.strip() for x in lines[i].strip("|").split("|")]
                if len(cells)!=len(headers): raise ValueError((i+1,headers,cells))
                row=dict(zip(headers,cells))
                add_chunk(chunks,byid,tags,sources,provenance,cid,h2,h3,h4,
                          "table_row",row,lines[i],[l,*intro],i+1,i+1)
                covered.add(i);i+=1
            stack=[];continue
        else:
            b=re.match(r"^(\s*)(?:[*-]|\d+\.)\s+(.*)",l)
            if b:
                kind="bullet";indent=len(b[1]);text=b[2]
                while stack and stack[-1][0]>=indent: stack.pop()
                parents=[s[1] for s in stack]
                stack.append((indent,text));i+=1
                # Non-bullet continuation remains with its parent fact.
                while i<refs_start and lines[i].strip() and not re.match(r"\s*(?:[*-]|\d+\.)\s+|[#|>]",lines[i]):
                    text += "\n"+lines[i];i+=1
            else:
                text=l;i+=1;stack=[]
                while i<refs_start and lines[i].strip() and not re.match(r"\s*(?:[*-]|\d+\.)\s+|[#|>]",lines[i]):
                    text += "\n"+lines[i];i+=1
                if not l.startswith(">") and refs(text): intro.append(text)
            # A field label is a key; other statements use an explicit generic key.
            m=re.match(r"(?:\*\*)?([^:\n]{1,60}?)(?:\*\*)?\s*:\s*(.*)",text,re.S)
            kv={"key":clean(m[1]) if m else "statement","value":m[2] if m else text}
        add_chunk(chunks,byid,tags,sources,provenance,cid,h2,h3,h4,kind,kv,text,
                  parents+intro,start+1,i)
        covered.update(range(start,i))
    ids=[c["chunk_id"] for c in chunks]
    assert len(ids)==len(set(ids)) and len(companies)==30
    used_refs={r for c in chunks for r in c["metadata"]["citation_ids"]}
    missing=sorted(used_refs-set(sources))
    if missing: raise ValueError(f"Unresolved citations: {missing}")
    expected={i for i,l in enumerate(lines[:refs_start]) if l.strip() and not l.startswith("#")
              and l.strip()!="---" and not (l.startswith("|") and i+1<refs_start and re.match(r"\|[-:| ]+\|",lines[i+1]))
              and not re.match(r"\|[-:| ]+\|",l)}
    assert not expected-covered, [(i+1,lines[i]) for i in expected-covered]
    summary={"company_count":30,"segment_company_counts":dict(Counter(c["segment"] for c in companies.values())),
             "chunk_count":len(chunks),"company_detail_chunk_count":sum(c["metadata"]["scope"]=="company" for c in chunks),
             "document_context_chunk_count":sum(c["metadata"]["scope"]!="company" for c in chunks),
             "source_count":len(sources),"unresolved_citations":missing,"content_coverage":"all non-heading source content before references",
             "source_as_of":date,"externally_reverified":False,
             "chunks_per_company":{c["name"]:len(c["chunk_ids"]) for c in companies.values()}}
    dump(out/"companies.json",{"schema_version":"1.0","source":provenance,"segments":SEGMENTS,"companies":companies})
    dump(out/"chunks.json",{"schema_version":"1.0","source":provenance,"chunks":{c["chunk_id"]:c for c in chunks}})
    (out/"chunks.jsonl").write_text("".join(json.dumps(c,ensure_ascii=False)+"\n" for c in chunks),encoding="utf-8")
    dump(out/"sources.json",sources);dump(out/"validation.json",summary)
    dump(out/"revenue_records.json",revenue_records(chunks,sources))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def add_chunk(chunks,byid,tags,sources,provenance,cid,h2,h3,h4,kind,kv,text,parents,start,end):
    path=[v for v in [h2,h3,h4] if v];cat=category(h4) if cid else "document_guidance"
    explicit=refs(text);inherited=sorted(set(refs("\n".join([h4,*parents])))-set(explicit))
    # Finance tables sometimes cite their audit source only in the following note.
    finance_table_source={"MOB":"MOB-1","BOS":"BOS-1","HA":"HA-1","RBL":"RBL-1","PAN":"PAN-1"}
    if cid and cat=="financial" and kind=="table_row" and not explicit and not inherited and cid in finance_table_source:
        inherited=[finance_table_source[cid]]
    citations=sorted(set(explicit+inherited));ctx="\n".join([h4,*parents,text])
    owners=[cid] if cid else entities(text,tags)
    # Company summary rows with exactly one named company support company filters too.
    owner=cid or (owners[0] if kind=="table_row" and len(owners)==1 else None)
    c=byid[owner] if owner else None
    cp=clean("\n".join(parents))
    taglist=[]
    if c:
        taglist=[f"company:{c['name']}",f"segment:{c['segment']}",
                 *[f"country:{x}" for x in c["country"]],
                 *[f"business_model:{x}" for x in c["business_models"]],
                 *[f"feature:{x}" for x in c["features"]],*[f"product:{x}" for x in c["products"]]]
    flags=markers(ctx)
    if not citations:flags.append("no_linked_external_citation")
    title_context=f"기업: {c['name']} ({', '.join(c['aliases'])}); 분류: {c['segment_label']}" if c else "문서 공통 가이드·교차 분석"
    # Concise labels supply semantic context; large tag lists remain in filter metadata.
    content=clean(text) if kind!="table_row" else "; ".join(f"{clean(k)}: {clean(v)}" for k,v in kv.items())
    embedding_text="\n".join(x for x in [title_context,"항목: "+" > ".join(path),
                             "상위 문맥: "+cp if cp else "",content] if x)
    token=hashlib.sha256((provenance["sha256"]+f"|{start}|{end}|"+text).encode()).hexdigest()[:16]
    chunk_id=f"{owner or 'DOC'}_{token}"
    metadata={"company_id":owner,"company_name":c["name"] if c else None,"company_ids":owners,
              "segment":c["segment"] if c else None,"country":c["country"] if c else [],
              "business_models":c["business_models"] if c else [],"features":c["features"] if c else [],
              "products":c["products"] if c else [],"tags":taglist,"category":cat,
              "scope":"company" if cid else "document_context","section_path":path,
              "source_line_start":start,"source_line_end":end,"citation_ids":citations,
              "explicit_citation_ids":explicit,"context_citation_ids":inherited,
              "source_grades":sorted({sources[r]["grade"] for r in citations if r in sources}),
              "literal_quality_markers":sorted(set(flags)),"document_as_of":provenance["document_as_of"],
              "eligibility_flags":c["eligibility_flags"] if c else [],"external_sources_verified":False}
    metadata["agent_routes"]={"overview":["technology","market","SWOT"],"technology":["technology","SWOT"],
      "technical_evidence":["technology","SWOT"],"funding":["financial","investment","SWOT"],
      "financial":["financial","investment","SWOT"],"commercialization":["technology","market","SWOT"],
      "risks":["technology","market","investment","SWOT"],"ip_ownership":["technology","investment"],
      "context":["investment","SWOT"],"document_guidance":["shared_guidance"]}[cat]
    chunk={"chunk_id":chunk_id,"kind":kind,"key_value":kv,"text_raw":text,
           "parent_context_raw":parents,"embedding_text":embedding_text,"metadata":metadata}
    chunks.append(chunk)
    if c:
        c["chunk_ids"].append(chunk_id)
        c["facts"].setdefault(cat,[]).append({"chunk_id":chunk_id,"key_value":kv,"text_raw":text,
                                           "parent_context_raw":parents,"citation_ids":citations,
                                           "source_lines":[start,end]})

def revenue_records(chunks,sources):
    result=[]
    for c in chunks:
        m=c['metadata'];row=c['key_value']
        if c['kind']!='table_row' or m['scope']!='company' or m['category']!='financial' or '매출' not in row:continue
        year=next((re.search(r'20\d{2}',clean(v))[0] for k,v in row.items() if k.startswith('연도') and re.search(r'20\d{2}',v)),None)
        raw=clean(row['매출']);amount=None;currency='USD' if '$' in raw else ('KRW' if '원' in raw or '억' in raw else None)
        unknown=bool(re.search(r'미공개|비공개|표시 없음|확인 안|매출 없음|^—$',raw))
        if not unknown:
            if '$' in raw:
                match=re.search(r'\$([\d,.]+)\s*([MB])?',raw)
                if match:amount=Decimal(match[1].replace(',',''))*{'M':Decimal(1000000),'B':Decimal(1000000000),None:Decimal(1)}[match[2]]
            else:
                # Exact won before rounded parenthesized amounts has priority.
                exact=re.search(r'(?<![\d.])([\d,]+)\s*원',raw)
                eok=re.search(r'([\d,.]+)\s*억',raw);man=re.search(r'([\d,.]+)\s*만',raw)
                if eok or man:
                    amount=(Decimal(eok[1].replace(',',''))*100000000 if eok else 0)+(Decimal(man[1].replace(',',''))*10000 if man else 0)
                if exact and not eok and not man: amount=Decimal(exact[1].replace(',',''))
                if exact and '(' in raw and raw.index('원')<raw.index('('):amount=Decimal(exact[1].replace(',',''))
        context=clean('\n'.join([*m['section_path'],*c['parent_context_raw'],str(row)]))
        forecast=bool(re.search(r'전망|목표|확정 실적 아님',context))
        unaudited=bool(re.search(r'감사받지 않',context))
        comparative=bool(re.search(r'비교 수치',context))
        grades=m['source_grades'];audited='공시' in grades and '감사' in context and not unaudited and not comparative
        metric_type='forecast' if forecast else ('actual' if audited else 'reported')
        if unknown:metric_type='unknown'
        result.append({'chunk_id':c['chunk_id'],'company_id':m['company_id'],'company_name':m['company_name'],
          'revenue_year':int(year) if year else None,'currency':currency,'amount':int(amount) if amount is not None and amount==int(amount) else (float(amount) if amount is not None else None),
          'amount_raw':raw,'amount_precision':'source_rounded' if amount is not None and re.search(r'억|약|M|B',raw) and not ('원 (' in raw or '원(' in raw) else ('source_exact' if amount is not None else 'unknown'),
          'metric_type':metric_type,'audit_status':'unaudited' if unaudited else ('comparative_audit_not_established' if comparative else ('audited_as_stated_in_document' if audited else 'not_established')),
          'accounting_basis_raw':context,'source_ids':m['citation_ids'],
          'source_urls':sorted({u for ref in m['citation_ids'] for u in sources[ref]['urls']}),
          'source_dates':sorted({sources[r]['source_date'] for r in m['citation_ids'] if sources[r]['source_date']}),
          'document_as_of':m['document_as_of'],'verified_at':None})
    # Exact amounts stated in source notes take precedence over rounded table cells.
    exact_note_amounts={('BOS',2025):'3,857,831,213',('RBL',2025):'32,021,696,412',('FUR',2025):'5,742,369,775'}
    for r in result:
        exact=exact_note_amounts.get((r['company_id'],r['revenue_year']))
        if exact:
            note=next(c for c in chunks if c['metadata']['company_id']==r['company_id'] and exact in c['text_raw'])
            r['amount']=int(exact.replace(',',''));r['amount_precision']='source_exact'
            r['exact_amount_note_chunk_id']=note['chunk_id'];r['exact_amount_note_raw']=note['text_raw']
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("--out",type=Path,default=ROOT/"data")
    a=p.parse_args();build(a.source,a.out)
