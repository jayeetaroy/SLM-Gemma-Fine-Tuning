#!/usr/bin/env python3
"""Train Gemma QLoRA with prompt tokens masked from loss."""
import argparse, json, os, platform, subprocess
from pathlib import Path
from common import chat_prompt, hf_token

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--config',default='configs/train.json'); ap.add_argument('--data-dir',default='data/processed'); ap.add_argument('--output-dir',required=True); ap.add_argument('--max-steps',type=int,default=-1); a=ap.parse_args()
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    cfg=json.loads(Path(a.config).read_text()); set_seed(cfg['seed']); token=hf_token(); tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],token=token); tokenizer.padding_side='right'
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    def load(name): return [json.loads(line) for line in (Path(a.data_dir)/f'{name}.jsonl').read_text().splitlines() if line.strip()]
    def encode(row):
        prompt=chat_prompt(tokenizer,row['question'],row['tools']); answer=json.dumps(row['target'],ensure_ascii=False)+tokenizer.eos_token
        p=tokenizer(prompt,add_special_tokens=False)['input_ids']; y=tokenizer(answer,add_special_tokens=False)['input_ids']; ids=(p+y)[:cfg['max_length']]
        labels=([-100]*len(p)+y)[:cfg['max_length']]
        if len(labels)!=len(ids): raise ValueError('label mask mismatch')
        return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels}
    train=Dataset.from_list([encode(x) for x in load('train')]); val=Dataset.from_list([encode(x) for x in load('validation')])
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    model=AutoModelForCausalLM.from_pretrained(cfg['base_model'],quantization_config=quant,device_map='auto',token=token); model=prepare_model_for_kbit_training(model); model.config.use_cache=False
    l=cfg['lora']; model=get_peft_model(model,LoraConfig(r=l['r'],lora_alpha=l['alpha'],lora_dropout=l['dropout'],target_modules=l['target_modules'],task_type='CAUSAL_LM'))
    t=cfg['training']; args=TrainingArguments(output_dir=a.output_dir,per_device_train_batch_size=t['per_device_train_batch_size'],per_device_eval_batch_size=1,gradient_accumulation_steps=t['gradient_accumulation_steps'],num_train_epochs=t['num_train_epochs'],learning_rate=t['learning_rate'],logging_steps=t['logging_steps'],save_strategy=t['save_strategy'],eval_strategy='epoch',gradient_checkpointing=t['gradient_checkpointing'],fp16=dtype==torch.float16,bf16=dtype==torch.bfloat16,report_to='none',seed=cfg['seed'],max_steps=a.max_steps)
    trainer=Trainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=DataCollatorForSeq2Seq(tokenizer,model=model,padding=True),processing_class=tokenizer)
    trainer.train(); trainer.save_model(a.output_dir); tokenizer.save_pretrained(a.output_dir)
    metadata={'config':cfg,'base_revision':model.config._name_or_path,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch':torch.__version__,'python':platform.python_version(),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'package_versions':{}}
    for pkg in ('transformers','peft','trl','bitsandbytes','accelerate','datasets'):
        try: metadata['package_versions'][pkg]=__import__(pkg).__version__
        except Exception: pass
    (Path(a.output_dir)/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
if __name__=='__main__': main()
