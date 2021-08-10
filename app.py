import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import Flask, request, Response, render_template, jsonify
import random
import time
import json
from  tabulate import tabulate

app = Flask(__name__, static_url_path='/static')

device = torch.device("cpu")
@app.route('/')
def index():
    return render_template('index.html')

from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("gpt2")

# %%
import torch
import string

from transformers import RobertaTokenizer, RobertaForMaskedLM
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaForMaskedLM.from_pretrained('roberta-base').eval()

top_k = 30


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
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'

    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx


def get_all_predictions(text_sentence, top_clean=30):
    # ========================= BERT =================================

    # ========================= ROBERTA =================================
    input_ids, mask_idx = encode(roberta_tokenizer, text_sentence, add_special_tokens=True)
    with torch.no_grad():
        predict = roberta_model(input_ids)[0]
    roberta = decode(roberta_tokenizer, predict[0, mask_idx, :].topk(top_k).indices.tolist(), top_clean)

    return {'roberta': roberta}



@app.route('/cool_form', methods=['GET', 'POST'])
def cool_form():
    if request.method == 'POST':
        # do stuff when the form is submitted

        # redirect to end the POST handling
        # the redirect can be to the same route or somewhere else
        return redirect(url_for('index'))

    # show the form, it wasn't submitted
    return render_template('cool_form.html')


@app.route('/predict', methods=["POST"])
def predict():
    start = time.time()
    if request.method == 'POST':
        text1 = request.form['rawtext']
        m = text1
        text = tokenizer.encode(text1)
        myinput, past = torch.tensor([text]), None

###

#bad words

        logits, past = model(myinput, past = past)
        logits = logits[0,-1]
        probabilities = torch.nn.functional.softmax(logits)
        best_logits, best_indices = logits.topk(780)
        best_words = [tokenizer.decode([idx.item()]) for idx in best_indices]
        text.append(best_indices[0].item())
        best_probabilities = probabilities[best_indices].tolist()
        for i in range(780):
            f = ('Generated {}: {}'.format(i, best_words[i]))
            print(f)
        #options = get_preds(text)
        #my_prediction = options
        #print(options)
        #print(options)
        table = [best_words[0:20]]
        p = (tabulate(table, tablefmt='html'))
        table = [best_words[20:40]]
        p2 = (tabulate(table, tablefmt='html'))
        table = [best_words[40:60]]
        p3 = (tabulate(table, tablefmt='html'))
        table = [best_words[60:80]]
        p4 = (tabulate(table, tablefmt='html'))
        table = [best_words[80:100]]
        p5 = (tabulate(table, tablefmt='html'))
        table = [best_words[100:120]]
        p6 = (tabulate(table, tablefmt='html'))
        table = [best_words[120:140]]
        p7 = (tabulate(table, tablefmt='html'))
        table = [best_words[140:160]]
        p8 = (tabulate(table, tablefmt='html'))
        table = [best_words[160:180]]
        p9 = (tabulate(table, tablefmt='html'))
        table = [best_words[180:200]]
        p10 = (tabulate(table, tablefmt='html'))
        table = [best_words[200:220]]
        p11 = (tabulate(table, tablefmt='html'))
        table = [best_words[220:240]]
        p12 = (tabulate(table, tablefmt='html'))
        table = [best_words[240:260]]
        p13 = (tabulate(table, tablefmt='html'))
        table = [best_words[260:280]]
        p14 = (tabulate(table, tablefmt='html'))
        table = [best_words[280:300]]
        p15 = (tabulate(table, tablefmt='html'))
        table = [best_words[300:320]]
        p16 = (tabulate(table, tablefmt='html'))
        table = [best_words[320:340]]
        p17 = (tabulate(table, tablefmt='html'))
        table = [best_words[340:360]]
        p18 = (tabulate(table, tablefmt='html'))
        table = [best_words[360:380]]
        p19 = (tabulate(table, tablefmt='html'))
        table = [best_words[380:400]]
        p20 = (tabulate(table, tablefmt='html'))
        table = [best_words[400:420]]
        p21 = (tabulate(table, tablefmt='html'))
        table = [best_words[420:440]]
        p15 = (tabulate(table, tablefmt='html'))
        table = [best_words[440:460]]
        p16 = (tabulate(table, tablefmt='html'))
        table = [best_words[460:480]]
        p17 = (tabulate(table, tablefmt='html'))
        table = [best_words[480:500]]
        p18 = (tabulate(table, tablefmt='html'))
        table = [best_words[500:520]]
        p19 = (tabulate(table, tablefmt='html'))
        table = [best_words[520:540]]
        p20 = (tabulate(table, tablefmt='html'))
        table = [best_words[540:560]]
        p21 = (tabulate(table, tablefmt='html'))
        table = [best_words[560:580]]
        p21 = (tabulate(table, tablefmt='html'))
        table = [best_words[580:600]]
        p22 = (tabulate(table, tablefmt='html'))
        table = [best_words[600:620]]
        p23 = (tabulate(table, tablefmt='html'))
        table = [best_words[620:640]]
        p24 = (tabulate(table, tablefmt='html'))
        table = [best_words[640:660]]
        p25 = (tabulate(table, tablefmt='html'))
        table = [best_words[660:680]]
        p26 = (tabulate(table, tablefmt='html'))
        table = [best_words[680:700]]
        p27 = (tabulate(table, tablefmt='html'))
        table = [best_words[700:720]]
        p28 = (tabulate(table, tablefmt='html'))
        table = [best_words[720:740]]
        p29 = (tabulate(table, tablefmt='html'))
        table = [best_words[740:760]]
        p30 = (tabulate(table, tablefmt='html'))
        table = [best_words[760:780]]
    end = time.time()
    final_time = '{:.2f}'.format((end-start))
    return render_template('index.html', m = text1, prediction=('{}'.format(p)), prediction2=('{}'.format(p2)), prediction3=('{}'.format(p3)),  prediction4=('{}'.format(p4)), prediction5=('{}'.format(p5)), prediction6=('{}'.format(p6)), prediction7=('{}'.format(p7)), prediction8=('{}'.format(p8)), prediction9=('{}'.format(p9)), prediction10=('{}'.format(p10)), prediction11=('{}'.format(p11)),  prediction12=('{}'.format(p12)), prediction13=('{}'.format(p13)), prediction14=('{}'.format(p14)), prediction15=('{}'.format(p15)), prediction16=('{}'.format(p16)),  prediction17=('{}'.format(p17)),  prediction18=('{}'.format(p18)),  prediction19=('{}'.format(p19)), prediction20=('{}'.format(p20)), prediction21=('{}'.format(p21)), prediction22=('{}'.format(p22)), prediction23=('{}'.format(p23)), prediction24=('{}'.format(p24)), prediction25=('{}'.format(p25)), prediction26=('{}'.format(p26)), prediction27=('{}'.format(p27)), prediction28=('{}'.format(p28)), prediction29=('{}'.format(p29)), prediction30=('{}'.format(p30)), final_time=final_time, text1=text1)
    #return render_template('index.html', prediction=('{}  | | | | |  Generated {}: {}'.format(text1, i, best_words[0:500])), final_time=final_time)


@app.route('/get_end_predictions', methods=['post'])
def get_prediction_eos():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        input_text += ' <mask>'
        top_k = request.json['top_k']
        res = get_all_predictions(input_text, top_clean=int(top_k))
        return app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')


@app.route('/get_mask_predictions', methods=['post'])
def get_prediction_mask():
    try:
        input_text = ' '.join(request.json['input_text'].split())
        top_k = request.json['top_k']
        res = get_all_predictions(input_text, top_clean=int(top_k))
        return app.response_class(response=json.dumps(res), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')

# Today, individuals worldwide are experiencing the loss of their basic right to privacy. 

#on account of

if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=80)
    app.run(debug=True, host='0.0.0.0', port=80, threaded=True)
