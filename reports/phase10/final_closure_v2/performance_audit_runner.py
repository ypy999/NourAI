from __future__ import annotations
import ctypes, hashlib, json, os, pathlib, platform, statistics, sys, time
from ctypes import wintypes
from datetime import datetime
from torch.utils.data import DataLoader
import torch

ROOT=pathlib.Path(r'D:\NourAI')
OUT=ROOT/'reports/phase10/final_closure_v2'
SPLITS=OUT/'splits_v2_verified'
CORPUS=ROOT/'data/processed/phase8/imported/tatoeba_cc0_multilingual_v4.jsonl'
sys.path.insert(0,str(ROOT))
from configs.configuration import load_config
from datasets.dataset import TokenSequenceDataset
from tokenizer.byte_tokenizer import ByteTokenizer

def windows_memory():
    info={'available':False}
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_=[('dwLength',ctypes.c_ulong),('dwMemoryLoad',ctypes.c_ulong),('ullTotalPhys',ctypes.c_ulonglong),('ullAvailPhys',ctypes.c_ulonglong),('ullTotalPageFile',ctypes.c_ulonglong),('ullAvailPageFile',ctypes.c_ulonglong),('ullTotalVirtual',ctypes.c_ulonglong),('ullAvailVirtual',ctypes.c_ulonglong),('ullAvailExtendedVirtual',ctypes.c_ulonglong)]
        m=MEMORYSTATUSEX();m.dwLength=ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        class PMC(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
        p=PMC();p.cb=ctypes.sizeof(p); psapi=ctypes.WinDLL('psapi',use_last_error=True);kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(PMC),wintypes.DWORD]; psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(p),p.cb): raise ctypes.WinError(ctypes.get_last_error())
        info={'available':True,'ram_total_bytes':int(m.ullTotalPhys),'ram_available_bytes':int(m.ullAvailPhys),'ram_used_percent':int(m.dwMemoryLoad),
              'process_working_set_bytes':int(p.WorkingSetSize),'process_peak_working_set_bytes':int(p.PeakWorkingSetSize),'process_pagefile_bytes':int(p.PagefileUsage)}
    except Exception as exc:
        info={'available':False,'error':type(exc).__name__+': '+str(exc)}
    return info

def hash_batch(batch):
    h=hashlib.sha256()
    for tensor in batch:
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()

def collect_batches(loader, max_batches=None):
    n_batches=0; n_tokens=0; latencies=[]; first_hash=None; shape_ok=True; shift_ok=True; dtype_ok=True; start=time.perf_counter()
    iterator=iter(loader)
    while True:
        t=time.perf_counter()
        try: batch=next(iterator)
        except StopIteration: break
        latencies.append(time.perf_counter()-t); n_batches+=1
        x,y=batch
        shape_ok &= x.ndim==2 and y.shape==x.shape and x.shape[1]==128
        dtype_ok &= x.dtype==torch.int64 and y.dtype==torch.int64
        if n_batches==1:
            first_hash=hash_batch(batch)
            shift_ok &= torch.equal(x[:,1:],y[:,:-1])
        n_tokens+=int(x.numel())
        if max_batches is not None and n_batches>=max_batches: break
    elapsed=time.perf_counter()-start
    return {'batches':n_batches,'input_tokens':n_tokens,'elapsed_seconds':elapsed,'tokens_per_second':n_tokens/elapsed if elapsed else 0,
            'mean_batch_seconds':statistics.mean(latencies) if latencies else 0,'median_batch_seconds':statistics.median(latencies) if latencies else 0,
            'p95_batch_seconds':sorted(latencies)[max(0,int(.95*len(latencies))-1)] if latencies else 0,
            'first_batch_sha256':first_hash,'shapes_valid':bool(shape_ok),'dtypes_valid':bool(dtype_ok),'next_token_shift_valid':bool(shift_ok)}

def main():
    started=time.perf_counter(); tokenizer=ByteTokenizer(); config=load_config(tokenizer=tokenizer); before=windows_memory()
    corpus_start=time.perf_counter(); records=0; chars=0; tokens=0; roundtrip_failures=0; invalid_ids=0; invalid_rows=0
    with CORPUS.open('r',encoding='utf-8',errors='strict') as stream:
        for line_no,line in enumerate(stream,1):
            try: row=json.loads(line)
            except Exception: invalid_rows+=1; continue
            text=row.get('text')
            if not isinstance(text,str): invalid_rows+=1; continue
            encoded=tokenizer.encode(text)
            records+=1; chars+=len(text); tokens+=len(encoded)
            if any(not isinstance(i,int) or i<0 or i>=tokenizer.vocab_size for i in encoded): invalid_ids+=1
            if tokenizer.decode(encoded,errors='strict')!=text: roundtrip_failures+=1
    corpus_elapsed=time.perf_counter()-corpus_start
    sample='العربية English français 🙂 123 ! '
    length_benchmarks=[]
    for requested in (32,256,2048,16384):
        text=(sample*((requested//len(sample))+1))[:requested]
        repeats=max(1,4000000//max(1,len(text)))
        t0=time.perf_counter(); n_tok=0
        for _ in range(repeats): n_tok+=len(tokenizer.encode(text))
        elapsed=time.perf_counter()-t0
        length_benchmarks.append({'characters_per_text':len(text),'repeats':repeats,'tokens':n_tok,'elapsed_seconds':elapsed,'tokens_per_second':n_tok/elapsed if elapsed else 0})
    dataset_results={}; datasets={}
    for split in ('train','validation','test'):
        t0=time.perf_counter(); ds=TokenSequenceDataset(SPLITS/f'{split}.jsonl',config,tokenizer); init_elapsed=time.perf_counter()-t0; datasets[split]=ds
        dataset_results[split]={'records_in_jsonl':sum(1 for _ in (SPLITS/f'{split}.jsonl').open('rb')),'dataset_sequences':len(ds),'token_count':ds.token_count,'initialization_seconds':init_elapsed,'tokenization_tokens_per_second':ds.token_count/init_elapsed if init_elapsed else 0}
    full_pass={}
    for split,ds in datasets.items():
        loader=DataLoader(ds,batch_size=config.batch_size,shuffle=False,num_workers=0)
        full_pass[split]=collect_batches(loader)
    train_ds=datasets['train']
    batch_sizes={}
    for batch_size in (1,16):
        batch_sizes[str(batch_size)]=collect_batches(DataLoader(train_ds,batch_size=batch_size,shuffle=False,num_workers=0),max_batches=128)
    # Deterministic shuffle order with a fixed seed, without updating model weights.
    def shuffled_hash(seed):
        gen=torch.Generator().manual_seed(seed)
        return collect_batches(DataLoader(train_ds,batch_size=8,shuffle=True,num_workers=0,generator=gen),max_batches=4)['first_batch_sha256']
    det_a=shuffled_hash(config.random_seed);det_b=shuffled_hash(config.random_seed)
    worker_result={'num_workers':1,'status':'NOT_RUN'}
    try:
        worker_result.update(collect_batches(DataLoader(train_ds,batch_size=8,shuffle=False,num_workers=1),max_batches=64))
        worker_result['status']='PASS'
    except Exception as exc:
        worker_result={'num_workers':1,'status':'FAILED','error_type':type(exc).__name__,'error':str(exc)[:1000]}
    after=windows_memory()
    for ds in datasets.values():
        try: ds._temp.cleanup()
        except Exception: pass
    result={'created_at_local':datetime.now().astimezone().isoformat(),'environment':{'python':sys.version,'executable':sys.executable,'os':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),
        'torch':torch.__version__,'torch_device':'cpu','cuda_used':False,'torch_cpu_threads':torch.get_num_threads(),'memory_before':before,'memory_after':after},
        'tokenizer':{'name':'ByteTokenizer','vocab_size':tokenizer.vocab_size,'corpus_path':str(CORPUS),'corpus_records':records,'characters':chars,'tokens':tokens,'elapsed_seconds':corpus_elapsed,
            'records_per_second':records/corpus_elapsed if corpus_elapsed else 0,'tokens_per_second':tokens/corpus_elapsed if corpus_elapsed else 0,'invalid_rows':invalid_rows,'invalid_token_ids':invalid_ids,'roundtrip_failures':roundtrip_failures,
            'length_benchmarks':length_benchmarks,'measurement_note':'Length benchmark uses deterministic multilingual synthetic strings; corpus throughput uses all actual imported records.'},
        'dataset':{'path':str(SPLITS),'splits':dataset_results,'note':'Each full JSONL split was read and byte-tokenized by the existing TokenSequenceDataset; no training was run.'},
        'dataloader':{'configured_batch_size':config.batch_size,'workers':{'full_corpus_worker_count':0,'alternate_worker_test':worker_result},'full_pass_configured_batch_size':full_pass,
            'alternate_batch_size_sampled':batch_sizes,'determinism':{'seed':config.random_seed,'first_batch_hash_run1':det_a,'first_batch_hash_run2':det_b,'repeatable':det_a==det_b},
            'note':'Configured batch size processed every sequence in each split; batch sizes 1/16 and worker=1 measurements use at most 128/64 batches.'},
        'total_elapsed_seconds':time.perf_counter()-started,'training_started':False}
    (OUT/'performance_final.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    md=['# CPU Performance Audit','',f"Run time: {result['created_at_local']}",f"Environment: {platform.platform()}, {platform.processor()}, {os.cpu_count()} logical CPUs, PyTorch {torch.__version__}; CPU only.",f"Physical RAM: {before.get('ram_total_bytes',0)/1024**3:.2f} GiB total; {before.get('ram_available_bytes',0)/1024**3:.2f} GiB available at start.",'',f"## Tokenizer",f"Actual corpus: {records:,} rows, {tokens:,} byte tokens in {corpus_elapsed:.2f}s ({tokens/corpus_elapsed:,.0f} tokens/s); round-trip failures {roundtrip_failures}; invalid IDs {invalid_ids}.",'','| Input length (chars) | Tokens/s |','|---:|---:|']
    for b in length_benchmarks:md.append(f"| {b['characters_per_text']} | {b['tokens_per_second']:,.0f} |")
    md+=['','## Dataset/DataLoader','',f"Configured batch size {config.batch_size}, workers 0; complete sequences traversed for all splits.",'','| Split | JSONL rows | Token cache tokens | Init seconds | Full batches | Full pass seconds | Tokens/s |','|---|---:|---:|---:|---:|---:|---:|']
    for split in ('train','validation','test'):
        d=dataset_results[split];p=full_pass[split];md.append(f"| {split} | {d['records_in_jsonl']:,} | {d['token_count']:,} | {d['initialization_seconds']:.2f} | {p['batches']:,} | {p['elapsed_seconds']:.2f} | {p['tokens_per_second']:,.0f} |")
    md+=['',f"DataLoader worker=1 sample: `{worker_result['status']}`; {worker_result.get('elapsed_seconds',0):.2f}s for {worker_result.get('batches',0)} batches. Worker=0 full pass is the production measurement.",f"Seeded shuffle first-batch repeatability: `{det_a==det_b}`.",f"Process peak working set at end: {after.get('process_peak_working_set_bytes',0)/1024**2:.1f} MiB; current: {after.get('process_working_set_bytes',0)/1024**2:.1f} MiB.",'','These are CPU measurements on this machine. They are data-input benchmarks, not training throughput or model-quality measurements.','']
    (OUT/'performance_final.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'status':'completed','corpus':result['tokenizer'],'dataset':dataset_results,'full_pass':full_pass,'batch_sizes':batch_sizes,'worker':worker_result,'deterministic':det_a==det_b,'memory_before':before,'memory_after':after,'total_seconds':result['total_elapsed_seconds']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
