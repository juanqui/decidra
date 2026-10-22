#!/usr/bin/env python3
"""NVFP4 GPTQ quantization for d3-turbo with prefix_bidir decision calibration."""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from safetensors import safe_open
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from compressed_tensors.quantization.quant_scheme import NVFP4, QuantizationScheme
from llmcompressor import oneshot
from llmcompressor.modifiers.gptq import GPTQModifier

MODEL = Path("/workspace/models/d3-turbo")
OUT = Path("/workspace/models/d3-turbo-nvfp4")
AUX = ["chat_template.jinja","config.json","decision_config.json","d3_compose.py","d3_engine.py","d3_fast.py","d3_format.py","d3_kernels.py","d3_runtime.py","d3_server.py","modeling_d3.py","pipeline_d3.py","preprocessor_config.json","requirements.txt","tokenizer.json","tokenizer_config.json","video_preprocessor_config.json","vocab.json","merges.txt","readout.safetensors"]
class BlockCalibration(torch.utils.data.Dataset):
    def __init__(self,path,tokenizer,limit):
        self.rows=[]
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            r=json.loads(line); ids=tokenizer(r["text"],add_special_tokens=False)["input_ids"]
            if len(ids)>2048: continue
            self.rows.append((ids,int(r["state_len"])))
            if len(self.rows)>=limit: break
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        ids,split=self.rows[i]; L=len(ids); m=torch.zeros(1,L,L,dtype=torch.bool)
        for q in range(L): m[0,q,:split if q<split else L]=True
        return {"input_ids":torch.tensor(ids,dtype=torch.long),"full_mask":m,"linear_mask":torch.ones(L,dtype=torch.long)}
def collate(batch):
    r=batch[0]; ids=r["input_ids"][None,:]; L=ids.shape[1]
    pos=torch.arange(L,dtype=torch.long)[None,None,:].expand(4,1,-1)
    return {"input_ids":ids,"attention_mask":{"full_attention":r["full_mask"][None],"linear_attention":r["linear_mask"][None]},"position_ids":pos}
def post(out):
    from safetensors.torch import load_file,save_file
    files=list(out.glob("*.safetensors")); print("safetensors",[(f.name,f.stat().st_size) for f in files],flush=True)
    # remove an accidental lm_head from any single-file save
    for f in files:
        try: st=load_file(str(f))
        except Exception as e: print("skip",f,e); continue
        if any(k.endswith("lm_head.weight") for k in st):
            for k in list(st):
                if k.endswith("lm_head.weight"): del st[k]
            save_file(st,str(f),metadata={"format":"pt"}); print("stripped lm_head",f,flush=True)
    for name in AUX:
        src=MODEL/name
        if src.exists() and not (out/name).exists(): shutil.copy(src,out/name)
def main():
    global MODEL
    ap=argparse.ArgumentParser(); ap.add_argument("--model",type=Path,default=MODEL); ap.add_argument("--samples",type=int,default=1024); ap.add_argument("--calib",type=Path,default=Path("/workspace/calib.jsonl")); ap.add_argument("--out",type=Path,default=OUT); ap.add_argument("--scope",choices=["gate_up","mlp","all"],default="gate_up"); ap.add_argument("--max-memory",default="88GiB"); args=ap.parse_args()
    MODEL=args.model
    tok=AutoTokenizer.from_pretrained(str(MODEL))
    ds=BlockCalibration(args.calib,tok,args.samples)
    loader=DataLoader(ds,batch_size=1,shuffle=False,collate_fn=collate)
    print("calibration rows",len(ds),flush=True)
    model, loading=Qwen3_5ForCausalLM.from_pretrained(str(MODEL),dtype=torch.bfloat16,attn_implementation="sdpa",device_map="auto",max_memory={0:args.max_memory,"cpu":"90GiB"},key_mapping={r"^language_model\.":"model."},output_loading_info=True)
    missing=[k for k in loading.get("missing_keys",[]) if k!="lm_head.weight"]
    if missing: raise SystemExit(f"missing weights {missing[:8]}")
    model.lm_head=torch.nn.Linear(model.config.hidden_size,1,bias=False)
    with safe_open(str(MODEL/"model-00001-of-00011.safetensors"),"pt") as h:
        probe=h.get_tensor("language_model.layers.0.mlp.gate_proj.weight")
    if not torch.equal(model.model.layers[0].mlp.gate_proj.weight.float().cpu(),probe.float()):
        raise SystemExit("layer0 weight mismatch")
    scheme=QuantizationScheme(targets=["Linear"],**NVFP4)
    ignore=["re:.*lm_head","re:.*visual.*","re:.*linear_attn.*","readout"]
    if args.scope!="all": ignore.append("re:.*self_attn.*")
    if args.scope=="gate_up": ignore.append("re:.*down_proj.*")
    recipe=GPTQModifier(config_groups={"group_0":scheme},ignore=ignore)
    oneshot(model=model,recipe=recipe,dataset=loader,tokenizer=tok,max_seq_length=2048,num_calibration_samples=len(ds))
    model.save_pretrained(str(args.out),save_compressed=True)
    post(args.out)
    print("done",args.out,flush=True)
if __name__=="__main__": main()
