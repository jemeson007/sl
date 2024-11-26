import pandas as pd

d = pd.read_json('fllbck.json')

tt = d.to_json(orient='records')[1:-1].replace('},{','}{')

with open("https://drive.google.com/file/d/1spo_XG35qGn-v-BVCyuz8LgvaJEmISkH/view?usp=sharing") as ff:
  print(ff.write(tt.corr())