import flask
from flask import request
from urllib.request import urlopen
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
import re
import threading
import numpy as nyp
from threading import Timer, Thread
import pandas as pd
import seaborn as ss
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


"""
w#w
"""

app = flask.Flask(__name__)
app.config["DEBUG"] = True
REREAD_ON_QUERY = True


@app.route("/", methods=["GET"])
def home():
  d = pd.read_json('fllbck.json')

  t = d.to_json(orient='records')[1:-1].replace('},{','}{')

  with open("https://drive.google.com/file/d/15DwcVN70F6KeXEEq6Gv8hqJhEsKSH9yo/view?usp=sharing") as ff:
	print(ff.write(t)
	

@app.route("/sp", methods=["GET"])
def sp():
  fllbck = {
    "user_id": "12345",
	"metrics": [
	  {
	   "date": "2024-11-22",
		"steps": 8500,
		"heart_rate": 75,
		"sleep_hours": 6.5,
		"hrv": 45
	   },
	  {
	  "date": "2024-11-21",
	  "steps": 9500,
	  "heart_rate": 72,
	  "sleep_hours": 7.2,
	  "hrv": 50
	  }
	 ]
	}

  ff = fllbck.to_list()
  
  print("variance", nyp.var(ff))
  
@app.route("/dy", methods=["GET"])
def dy():
	  
  d = pd.read_json('fllbck.json')

  d.head()

  ss.set(style="whitegrid")

  plt.figure(figsize=(12, 6))  
  sns.lineplot(data=d, x='user_id', y='metrics', label='Downtime', color='blue')
	 
  # Adding labels and title
  plt.xlabel('user_id')
  plt.ylabel('metrics')
  plt.title('Analytic')
	 
  plt.show()

@app.route("/nl", methods=["GET"])
def nl():
  model_np = "TheBloke/Llama-2-7b-Chat-GPTQ"
  model = AutoModelForCausalLM.from_pretrained(model_np, device_map="auto", trust_remote_code=False, revision="main")

  tokenizer = AutoTokenizer.from_pretrained(model_np, use_fast = True)
  usr = pd.read_json('usr_jnl.json')
  nt = pd.Series(usr, index=["journal_entries"["entry"]])
  prompt = nt
  prompt_template = f'''[INST] <<SYS>> "Your recent journal entries reflect sadness. Would you like some tips for managing negative emotions?" <</SYS>>{prompt}[/INST] '''

  print('En route determinant:')

  input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
  output = model.generate(inputs=input_ids, temperature=0.8, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=256)
  print(tokenizer.decode(output[0]))
  


@app.route("/gg", methods=["GET"])
def gg():
  d = pd.read_json('fllbck.json')

  tt = d.to_json(orient='records')[1:-1].replace('},{','}{')

  with open("https://drive.google.com/file/d/1spo_XG35qGn-v-BVCyuz8LgvaJEmISkH/view?usp=sharing") as ff:
    print(ff.write(tt.corr())

def test_home():
    assert home() == True

app.run()