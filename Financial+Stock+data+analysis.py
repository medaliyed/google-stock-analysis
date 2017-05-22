
# coding: utf-8

# In[1]:

# 1) what was the change in price of the stock?
# 2) the daily return of the stock average?
# 3) the moving avg of the various stocks?
# 4) the correlation betwenn diffrent stocks  closing prices?
# 5) the correlation betwenn diffrent stocks daily returns?
# 6) the value do we put at risk by investing in a particular stock?
# 7) how can attempt to predict future stock behavior?


# In[2]:

import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[3]:

sb.set_style('whitegrid')
get_ipython().magic('matplotlib inline')
from pandas_datareader import data, wb


# In[4]:

from datetime import datetime


# In[5]:

from __future__ import division


# In[6]:

tech = ['AAPL','GOOG','MSFT','AMZN']


# In[7]:

end = datetime.now()


# In[8]:

strt = datetime(end.year-1, end.month, end.day)


# In[9]:

import pandas_datareader as pdr


# In[10]:

for stk in tech:
    globals()[stk] = pdr.DataReader(stk,'google',strt,end)


# In[11]:

AAPL.head()


# In[12]:

GOOG.head()


# In[13]:

GOOG.describe()


# In[14]:

GOOG.info()


# In[15]:

GOOG['Close'].plot(legend=True,figsize=(10,4))


# In[16]:

GOOG['Volume'].plot(legend=True,figsize=(10,4))


# In[17]:

GOOG['Open'].plot(legend=True,figsize=(10,4))


# In[18]:

mday = [10,20,50]
for mav in mday:
    columname = "MA %s days" %(str(mav))
    GOOG[columname] =pd.rolling_mean(GOOG['Close'],mav)


# In[19]:

GOOG[['Close','MA of 10 days','MA of 20 days','MA of 50 days']].plot(subplots=False,figsize=(10,4))


# In[20]:

GOOG['daily return'] = GOOG['Close'].pct_change()
GOOG['daily return'].plot(figsize=(10,4),legend=True,linestyle='--',marker='o')


# In[21]:

sb.distplot(GOOG['daily return'].dropna(),bins=100)


# In[22]:

GOOG['daily return'].hist()


# In[23]:

cldf = pdr.DataReader(tech,'google',strt,end)['Close']


# In[24]:

cldf.head()


# In[25]:

techrets = cldf.pct_change()


# In[26]:

techrets.head()


# In[27]:

sb.jointplot('GOOG','GOOG',techrets,kind='scatter',color='seagreen')


# In[28]:

sb.jointplot('GOOG','AAPL',techrets,kind='scatter',color='seagreen')


# In[29]:

sb.pairplot(techrets.dropna())


# In[30]:

retfig = sb.PairGrid(techrets.dropna())

retfig.map_upper(plt.scatter,color='purple')

retfig.map_lower(sb.kdeplot,cmap='cool_d')

retfig.map_diag(plt.hist,bins=30)


# In[31]:

retfig = sb.PairGrid(cldf)

retfig.map_upper(plt.scatter,color='purple')

retfig.map_lower(sb.kdeplot,cmap='cool_d')

retfig.map_diag(plt.hist,bins=30)


# In[32]:

sb.linearmodels.corrplot(techrets.dropna(),annot=True)


# In[33]:

sb.linearmodels.corrplot(cldf.dropna(),annot=True)


# In[34]:

#analysing the risk of the stock


# In[35]:

rets = techrets.dropna()


# In[36]:

area = np.pi*20
plt.scatter(rets.mean(),rets.std(),s=area)
plt.xlabel('expected return')
plt.ylabel('risk')


for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))


# In[37]:

# Value at risk using the "bootstrap" method
sb.distplot(GOOG['daily return'].dropna(),bins=100,color='purple')


# In[38]:

# The 0.05 empirical quantile of daily returns
rets['GOOG'].quantile(0.05)


# In[39]:

#Value at Risk using the Monte Carlo method


# In[40]:

days = 365
# delta
dt = 1/days
#(drift)
mu = rets.mean()['GOOG']
# average return
sigma = rets.std()['GOOG']


# In[52]:

def stock_monte_carlo(start_price,days,mu,sigma):
    ''' This function takes in starting stock price, days of simulation,mu,sigma, and returns simulated price array'''
    
    # price array
    price = np.zeros(days)
    price[0] = start_price
    # Schok and Drift
    shock = np.zeros(days)
    drift = np.zeros(days)
    
    # Run price array for number of days
    for x in range(1,days):
        
        # Calculate Schock
        shock[x] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        # Calculate Drift
        drift[x] = mu * dt
        # Calculate Price
        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price


# In[42]:

GOOG.head()


# In[53]:

start_price = 706.53

for run in range(1,100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))
    
    
plt.xlabel("Days")
plt.ylabel("Price")  
plt.title('Monte Carlo Analysis for Google')


# In[54]:

runs = 10000
# Create an empty matrix to hold the end price data
simulations = np.zeros(runs)

# Set the print options of numpy to only display 0-5 points from an array to suppress output
np.set_printoptions(threshold=5)

for run in  range(runs):    
    # Set the simulation data point as the last stock price for that run
    simulations[run] = stock_monte_carlo(start_price,days,mu,sigma)[days-1];


# In[55]:

# Now we'lll define q as the 1% empirical qunatile, this basically means that 99% of the values should fall between here
q = np.percentile(simulations, 1)
    
# Now let's plot the distribution of the end prices
plt.hist(simulations,bins=200)

# Using plt.figtext to fill in some additional information onto the plot

# Starting Price
plt.figtext(0.6, 0.8, s="Start price: $%.2f" %start_price)
# Mean ending price
plt.figtext(0.6, 0.7, "Mean final price: $%.2f" % simulations.mean())

# Variance of the price (within 99% confidence interval)
plt.figtext(0.6, 0.6, "VaR(0.99): $%.2f" % (start_price - q,))

# Display 1% quantile
plt.figtext(0.15, 0.6, "q(0.99): $%.2f" % q)

# Plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

# Title
plt.title(u"Final price distribution for Google Stock after %s days" % days, weight='bold');


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



