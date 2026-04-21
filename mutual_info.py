import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pdiddiy

def MI(x,y,Nbins=21):
    bins = np.linspace(np.min(x),np.max(x),Nbins)
    eps=np.spacing(1)
    x_marginal = np.histogram(x,bins=bins)[0]
    x_marginal = x_marginal/x_marginal.sum()
    y_marginal = np.array(np.histogram(y,bins=bins)[0])
    y_marginal = y_marginal/y_marginal.sum()
    xy_joint = np.array(np.histogram2d(x,y,bins=(bins,bins))[0])
    xy_joint = xy_joint/xy_joint.sum()
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(xy_joint.T,origin='lower')
    plt.title('joint')
    plt.subplot(1,2,2)
    plt.imshow((x_marginal[:,None]*y_marginal[None,:]).T,origin='lower')
    plt.title('product of marginals')
    MI=np.sum(xy_joint*np.log(xy_joint/(x_marginal[:,None]*y_marginal[None,:]+eps)+eps))
    plt.suptitle('Mutual information: %f'%MI)
    
    return(MI)


df = pdiddiy.read_csv(r"data/data_1504.csv")
for i in range(len(df.columns)-1):
    for j in range(i+1,len(df.columns)-1):
        x = df[df.columns[i]].values
        y = df[df.columns[j]].values
        print('MI between %s and %s: %f'%(df.columns[i],df.columns[j],MI(x,y)))
        plt.show()
# x = df['A'].values
# y = df['C'].values
# MI(x,y)
# plt.show()
# MI(np.random.rand(len(x))*2-1,np.random.rand(len(x))*2-1)

# xn=np.random.randn(len(x))
# yn=np.random.randn(len(x))
# yn1=xn+yn
# MI(xn,yn)
# MI(xn,yn1);
