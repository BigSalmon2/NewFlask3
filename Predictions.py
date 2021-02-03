#@title Predictions:
import torch
import string
top_k = 100
def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])
def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx
def get_all_predictions(text_sentence, top_clean=5):
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence)
    with torch.no_grad():
        predict = xlnet_model(input_ids)[0]
    xlnet = decode(xlnet_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = bart_model(input_ids)[0]
    bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = electra_model(input_ids)[0]
    electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return {'bert': bert,
            'xlnet': xlnet,
            'xlm': xlm,
            'bart': bart,
            'electra': electra,
            'roberta': roberta}

import torch
import string
top_k = 100
def decode(tokenizer, pred_idx, top_clean):
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return '\n'.join(tokens[:top_clean])
def encode(tokenizer, text_sentence, add_special_tokens=True):
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx
def get_all_predictions(text_sentence, top_clean=5):
    print(text_sentence)
    input_ids, mask_idx = encode(bert_tokenizer, text_sentence)
    with torch.no_grad():
        predict = bert_model(input_ids)[0]
    bert = decode(bert_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(xlnet_tokenizer, text_sentence)
    with torch.no_grad():
        predict = xlnet_model(input_ids)[0]
    xlnet = decode(xlnet_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(xlmroberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = xlmroberta_model(input_ids)[0]
    xlm = decode(xlmroberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(bart_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = bart_model(input_ids)[0]
    bart = decode(bart_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(electra_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = electra_model(input_ids)[0]
    electra = decode(electra_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)
    return {'bert': bert,
            'xlnet': xlnet,
            'xlm': xlm,
            'bart': bart,
            'electra': electra,
            'roberta': roberta}
def get_all_predictions2(text_sentence, top_clean=5):

    text = tokenizer.encode(text_sentence)
    myinput, past = torch.tensor([text]), None
    logits, past = model(myinput, past = past)
    logits = logits[0,-1]
    probabilities = torch.nn.functional.softmax(logits)
    best_logits, best_indices = logits.topk(100)
    best_words = [tokenizer.decode([idx.item()]) for idx in best_indices]
    text.append(best_indices[0].item())
    best_probabilities = probabilities[best_indices].tolist()
    xlnet = best_words[0:100]
    xlnet = xlnet

    text = tokenizer2.encode(text_sentence)
    myinput, past = torch.tensor([text]), None
    logits, past = model2(myinput, past = past)
    logits = logits[0,-1]
    probabilities = torch.nn.functional.softmax(logits)
    best_logits, best_indices = logits.topk(100)
    best_words = [tokenizer2.decode([idx.item()]) for idx in best_indices]
    text.append(best_indices[0].item())
    best_probabilities = probabilities[best_indices].tolist()
    bart = best_words[0:100]
    bart = bart
    text = tokenizer3.encode(text_sentence)

    myinput, past = torch.tensor([text]), None
    logits, past = model3(myinput, past = past)
    logits = logits[0,-1]
    probabilities = torch.nn.functional.softmax(logits)
    best_logits, best_indices = logits.topk(100)
    best_words = [tokenizer3.decode([idx.item()]) for idx in best_indices]
    text.append(best_indices[0].item())
    best_probabilities = probabilities[best_indices].tolist()
    bert = best_words[0:100]
    bert = bert

    text = tokenizer4.encode(text_sentence)
    myinput, past = torch.tensor([text]), None
    logits, past = model4(myinput, past = past)
    logits = logits[0,-1]
    probabilities = torch.nn.functional.softmax(logits)
    best_logits, best_indices = logits.topk(100)
    best_words = [tokenizer4.decode([idx.item()]) for idx in best_indices]
    text.append(best_indices[0].item())
    best_probabilities = probabilities[best_indices].tolist()
    xlm = best_words[0:100]
    xlm = xlm

    text = tokenizer5.encode(text_sentence)
    myinput, past = torch.tensor([text]), None
    logits, past = model5(myinput, past = past)
    logits = logits[0,-1]
    probabilities = torch.nn.functional.softmax(logits)
    best_logits, best_indices = logits.topk(100)
    best_words = [tokenizer5.decode([idx.item()]) for idx in best_indices]
    text.append(best_indices[0].item())
    best_probabilities = probabilities[best_indices].tolist()
    roberta = best_words[0:100]
    roberta = roberta


    text = tokenizer6.encode(text_sentence)
    myinput, past = torch.tensor([text]), None
    logits, past = model6(myinput, past = past)
    logits = logits[0,-1]
    probabilities = torch.nn.functional.softmax(logits)
    best_logits, best_indices = logits.topk(100)
    best_words = [tokenizer6.decode([idx.item()]) for idx in best_indices]
    text.append(best_indices[0].item())
    best_probabilities = probabilities[best_indices].tolist()
    electra = best_words[0:100]
    electra = electra



    return {'bert': bert, 'xlnet': xlnet, 'bart': bart, 'xlm': xlm, 'roberta': roberta, 'electra': electra,}
