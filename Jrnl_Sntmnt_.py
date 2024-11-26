from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

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