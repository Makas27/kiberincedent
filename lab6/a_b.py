# Packages imports
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.stats.api as sms
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

effect_size = sms.proportion_effectsize(0.13, 0.15)
required_n = sms.NormalIndPower().solve_power(effect_size, power=0.8, alpha=0.05, ratio=1)                                                  
required_n = ceil(required_n)                        
#print(required_n)
raw_data = pd.read_csv('ab_data.csv')
#print(raw_data.head())
session_counts = raw_data['user_id'].value_counts(ascending=False)
multi_users = session_counts[session_counts > 1].count()
#print(multi_users)
users_to_drop = session_counts[session_counts > 1].index
raw_data = raw_data[~raw_data['user_id'].isin(users_to_drop)]
control_sample = raw_data[raw_data['group'] == 'control'].sample(n=required_n, random_state=22)
treatment_sample = raw_data[raw_data['group'] == 'treatment'].sample(n=required_n, random_state=22)
ab_test = pd.concat([control_sample, treatment_sample], axis=0)
ab_test.reset_index(drop=True, inplace=True)
#print(ab_test)
#print(ab_test.info())
#print(ab_test['group'].value_counts())
conversion_rates = ab_test.groupby('group')['converted']
a = lambda x: np.std(x, ddof=0)
b = lambda x: stats.sem(x, ddof=0)            
conversion_rates = conversion_rates.agg([np.mean, a, b])
conversion_rates.columns = ['conversion_rate', 'deviation', 'error']
print(conversion_rates)
plt.figure(figsize=(8,6))
sns.barplot(x=ab_test['group'], y=ab_test['converted'], ci=False)
plt.title('Conversion rate by group', pad=20)
plt.xlabel('Group', labelpad=15)
plt.ylabel('Converted (proportion)', labelpad=15)
plt.show()