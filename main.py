import numpy as np
import pandas as pd
from typing import List
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from tqdm import tqdm
import yfinance as yf
from pypfopt.objective_functions import L2_reg
from pypfopt.discrete_allocation import DiscreteAllocation,get_latest_prices
from datetime import datetime

# daily_price_history has to at least have a column, called 'Close'
def yf_retrieve_data(tickers: List[str]):
    dataframes = []
    names = []
    for ticker_name in tqdm(tickers):
        history = yf.download(ticker_name, r'2020-05-10', datetime.today().strftime('%Y-%m-%d'))
        history = history.reset_index()
        if history.empty:
            continue
        if history.isnull().any(axis=1).iloc[0]:  # the first row can have NaNs
            history = history.iloc[1:]
        if history.shape[0] != 987:
            continue
        assert not history.isnull().any(axis=None), f'history has NaNs in {ticker_name}'
        if len(history['Close'].tolist()) != 987:
            continue
        dataframes.append(history['Close'].values)
        names.append(ticker_name)

    return dataframes, names, history['Date']


stocks = {'BKNG', 'AAPL', 'MSFT', 'GOOG', 'AMZN', 'IDXX', 'INTC', 'VOO', 'MDT', 'CB', 'COF',
          'XOM', 'BA', 'KO', 'HPQ', 'JNJ', 'AC', 'LLY', 'CVX', 'PFE', 'NKE', 'CRM', 'MS', 'RTX', 'LMT', 'CVS', 'MSI',
          'COF', 'CLX', 'CPB', 'J', 'DGX', 'MGA', 'TXT', 'BIO', 'DIS', 'T', 'LUV', 'MCD', 'GE', 'ALL', 'BA', 'CAT',
          'CMG', 'CSCO', 'COST', 'CTRA', 'CMI', 'DVA', 'DE', 'ECL', 'HSY', 'K', 'META','DGII','FBP','GFI','GPK',
              'EE','NGLOY','LZB','MRTN','RDWR','TGS''STER','FDUS','PTVE','UTI','FFBC',
           'GLAD','TBBK','WAFD','HZO','TTGT','ALEX','AORT','VBTX','DRQ''NGVT','LSEA','PCRFY',
           'SPOK','RUSHA','ISPR','CION','MEG','WSFS','CSTL','SPTN','HNI','HMN','SOXQ'}

# stocks = ['META,VOO']
daily_dataframes, names,dates = yf_retrieve_data(stocks)
df_stocks = pd.DataFrame(np.asarray(daily_dataframes).T)
df_stocks.columns = names
df_stocks.set_index(dates,inplace=True)
print(df_stocks)
mu = expected_returns.ema_historical_return(df_stocks)
Sigma = risk_models.risk_matrix(df_stocks,method = 'ledoit_wolf_single_factor')
# Max Sharpe Ratio - Tangent to the EF
ef = EfficientFrontier(mu, Sigma, weight_bounds=(0, 1))  # weight bounds in negative allows shorting of stocks
ef.add_objective(L2_reg, gamma=.005)
sharpe_pfolio = ef.max_sharpe()  # May use add objective to ensure minimum zero weighting to individual stocks
sharpe_pwt = ef.clean_weights()
print(ef.portfolio_performance(verbose=True))

da = DiscreteAllocation(sharpe_pwt, df_stocks.iloc[-1,:], total_portfolio_value=33_000)
allocation, leftover = da.lp_portfolio(reinvest=False)
print("Discrete allocation:", allocation)
pd.DataFrame([[i,allocation[i]]for i in allocation]).to_csv('invest.csv')
print("Funds remaining: ${:.2f}".format(leftover))