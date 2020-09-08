from flask_ngrok import run_with_ngrok
from flask import Flask, render_template, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from flask import request
import random
import time

app = Flask(__name__, root_path='/content/GLPAPP')
app.config['SECRET_KEY'] = '9bad6913d4358ac1395c5c94370ed090'
run_with_ngrok(app)
print(app.root_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    start = time.time()
    if request.method == 'POST':
        text1 = request.form['rawtext']
        text = tokenizer.encode(text1)
        myinput, past = torch.tensor([text]), None

###
        logits, past = model(myinput, past = past)
        logits = logits[0,-1]
        probabilities = torch.nn.functional.softmax(logits)
        best_logits, best_indices = logits.topk(540)
        best_words = [tokenizer.decode([idx.item()]) for idx in best_indices]
        text.append(best_indices[0].item())
        best_probabilities = probabilities[best_indices].tolist()
        for i in range(540):
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
    return render_template('index.html', prediction=('{}'.format(p)), prediction2=('{}'.format(p2)), prediction3=('{}'.format(p3)),  prediction4=('{}'.format(p4)), prediction5=('{}'.format(p5)), prediction6=('{}'.format(p6)), prediction7=('{}'.format(p7)), prediction8=('{}'.format(p8)), prediction9=('{}'.format(p9)), prediction10=('{}'.format(p10)), prediction11=('{}'.format(p11)),  prediction12=('{}'.format(p12)), prediction13=('{}'.format(p13)), prediction14=('{}'.format(p14)), prediction15=('{}'.format(p15)), prediction16=('{}'.format(p16)),  prediction17=('{}'.format(p17)),  prediction18=('{}'.format(p18)),  prediction19=('{}'.format(p19)), prediction20=('{}'.format(p20)), prediction21=('{}'.format(p21)), prediction22=('{}'.format(p22)), prediction23=('{}'.format(p23)), prediction24=('{}'.format(p24)), prediction25=('{}'.format(p25)), prediction26=('{}'.format(p26)), prediction27=('{}'.format(p27)), prediction28=('{}'.format(p28)), prediction29=('{}'.format(p29)), prediction30=('{}'.format(p30)), final_time=final_time, text1=text1)
    #return render_template('index.html', prediction=('{}  | | | | |  Generated {}: {}'.format(text1, i, best_words[0:500])), final_time=final_time)



if __name__ == '__main__':
    app.run()
