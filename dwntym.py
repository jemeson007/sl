import pandas as pd
import numpy as np
import seaborn as ss
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


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