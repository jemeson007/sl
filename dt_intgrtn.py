import pandas as pd

d = pd.read_json('fllbck.json')

t = d.to_json(orient='records')[1:-1].replace('},{','}{')

with open("https://drive.google.com/file/d/15DwcVN70F6KeXEEq6Gv8hqJhEsKSH9yo/view?usp=sharing") as ff:
  print(ff.write(t)