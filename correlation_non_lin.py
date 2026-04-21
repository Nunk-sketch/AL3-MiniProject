import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
if os.name == "nt":  # Windows
    df = pd.read_csv("data\data_1504.csv")
else:  # macOS/Linux
    df = pd.read_csv("data/data_1504.csv")


omega=np.random.rand(5000)*6*np.pi

x=np.sin(omega)+0.1*np.random.randn(len(omega))
y=np.cos(omega)+0.1*np.random.randn(len(omega))

plt.figure();
plt.plot(omega,x,'.');
plt.plot(omega,y,'.'); 
plt.legend(('x','y'));
plt.xlabel(r'$\omega$');
print('Correlation between x and y: %.2f'%np.corrcoef(x,y)[0,1]);
plt.figure();
sns.jointplot(x=x,y=y,kind='kde');