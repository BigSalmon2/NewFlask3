# Loading model dependencies
import numpy as np
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer

# Downloading the model
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelWithLMHead.from_pretrained('gpt2')


def get_preds(text, length=200, options=3, temp=0.7):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    generated = model.generate(input_ids=input_ids, max_length=length, min_length=int(length/1.05), do_sample=True, num_return_sequences=options, temperature=temp, top_p=0.92, top_k=50, no_repeat_ngram_size=2)
    resulting_string = [tokenizer.decode(generated[i], skip_special_tokens=True) for i in range(options)]
    return resulting_string
