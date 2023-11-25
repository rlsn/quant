import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.optimize import brentq
from scipy.stats import norm

from futu import *
import time

def FV(npv,r,t):
    return npv*(1+r/100)**t
def HPR(npv,fv,t):
    return ((fv/npv)**(1/t)-1)*100

class OptionPosition(object):
    def __init__(self, option_type, strike_price, quantity, cost_price, 
                 contract_size=500, implied_volatility=20.,expiry_date_distance=30):
        
        
        assert option_type in ["CALL", "PUT"]
        self.option_type = option_type # call/put
        self.strike_price = strike_price
        self.quantity = quantity
        self.cost_price = cost_price 
        
        self.contract_size = contract_size
        self.implied_volatility = implied_volatility
        self.expiry_date_distance = expiry_date_distance
       
    @classmethod
    def from_series(cls,s):
        opt = cls(s['option_type'], 
                  s['strike_price'], 
                  s['qty'], 
                  s['cost_price'],
                  s['option_contract_size'],
                  s['option_implied_volatility'],
                  s['option_expiry_date_distance'])
        return opt

    @classmethod
    def from_df(cls, df):
        position_objs = []
        for idx, pos in df.iterrows():
            opt = cls.from_series(pos)
            position_objs.append(opt)
        return position_objs
        
    def __repr__(self):
        return f"{self.option_type}{self.strike_price}/{int(self.quantity)}@{self.cost_price}"
        
        
    def ytm_function(self):
        def func(x):
            if self.option_type=="CALL":
                y = x-self.strike_price-self.cost_price
                y = np.where(x <= self.strike_price, -self.cost_price, y)
            elif self.option_type=="PUT":
                y = self.strike_price-x-self.cost_price
                y = np.where(x >= self.strike_price, -self.cost_price, y)
        
            y*=self.quantity*self.contract_size
            return y
        return func
    
    def realized_function(self): 
        def func(x,delta_t=0):
            theoretical_price = black_scholes(x,self.strike_price,self.option_type,
                                      self.expiry_date_distance+delta_t,self.implied_volatility)
            y = self.quantity * (theoretical_price - self.cost_price) * self.contract_size
            return y
        return func

    def ytm_curve(self,x):            
        y = self.return_function()(x)
        return y
    
    def realized_curve(self,x):
        y = self.realized_return_function()(x)
        return y
    
def find_roots(options):
    prices = sorted([opt.strike_price for opt in options])
    prices = [0]+prices+[prices[-1]*2]
    
    func =lambda x: sum([opt.ytm_function()(x) for opt in options]) 
    
    roots = []
    for i in range(len(prices)-1):
        try:
            r= brentq(func,prices[i],prices[i+1])
            roots.append(np.round(r,2))
        except ValueError:
            continue
    return roots

def draw_roots(roots,fig=None):
    if fig is None:
        fig, ax = plt.subplots(figsize=(6,4))
    for r in roots:
        plt.text(r,0, f"({r}, 0)",size=10)
        plt.plot(r,0,'x',markersize=6,color='grey')


def draw_curve(function, xstart, xend, intv, poi=[],show_min=False,show_max=False,
               fig=None,label="0",zero_line=False):
    X=np.arange(xstart,xend,intv)
    Y=function(X)

    if fig is None:
        fig, ax = plt.subplots(figsize=(6,4))
    if zero_line:
        plt.plot(X,np.zeros(X.shape),color='grey')
        
    plt.plot(X,Y,label=label)
    
    if show_max:
        plt.text(X[np.argmax(Y)],np.max(Y), f"({X[np.argmax(Y)]}, {np.max(Y)})", size=10)
        plt.plot(X[np.argmax(Y)],np.max(Y),'^',markersize=6,color='red')
        
    if show_min:
        plt.text(X[np.argmin(Y)],np.min(Y), f"({X[np.argmin(Y)]}, {np.min(Y)})", size=10)
        plt.plot(X[np.argmin(Y)],np.min(Y),'v',markersize=6,color='green')
        
    for x in poi:
        plt.plot(x,function(x),'x',markersize=8,color = 'black')
        plt.text(x,function(x), f"({x}, {function(x)})", size=10) 


def position_analysis(position_objs, current_stock_price):
    ytm_func =lambda x: sum([opt.ytm_function()(x) for opt in position_objs]) 
    realized_func =lambda x,dt=0: sum([opt.realized_function()(x,dt) for opt in position_objs]) 

    xstart=65
    xend=110
    intv=0.5
    fig, ax = plt.subplots(figsize=(6,4))

    draw_curve(ytm_func,xstart,xend,intv,show_max=True,show_min=True,fig=fig,label="YtM",zero_line=True)
    draw_curve(realized_func,xstart,xend,intv,show_max=True,fig=fig,label="Realized")
    draw_roots(find_roots(position_objs),fig=fig)
    ax.grid()
    ax.legend()

    ytm=ytm_func(current_stock_price)
    realized=realized_func(current_stock_price)
    unrealized=ytm-realized
    
    #greeks
    d = current_stock_price*0.005
    Delta = lambda x,d: (realized_func(x+d)-realized_func(x-d))/d/2
    delta = Delta(current_stock_price,d)
    theta = realized_func(current_stock_price)-realized_func(current_stock_price,-1)
    gamma = (Delta(current_stock_price+d*10,d)-Delta(current_stock_price-d*10,d))/d/2
    
    
    analysis = {
        "ytm":ytm,
        "realized":realized,
        "unrealized":unrealized,
        "delta": delta,
        "theta":theta,
        "gamma":gamma
    }
    plt.plot(current_stock_price, realized,'o',markersize=6,color='green')

    return analysis
    
def get_price(quote_ctx, code):
    ret, dat = quote_ctx.get_market_snapshot([code])
    if ret == RET_OK:
        return dat['last_price'][0]
    else:
        print('error:', dat)
    return 

def get_options(quote_ctx, code="HK.09988",date="2023-07-28"):
    filter1 = OptionDataFilter()
    filter1.delta_min = -0.95
    filter1.delta_max = 0.95

    ret2, data2 = quote_ctx.get_option_chain(code=code, start=date, end=date, data_filter=filter1)
    if ret2 == RET_OK:
        result = data2[['code', 'strike_price', 'option_type']]
                    
        ret3, data3 = quote_ctx.get_market_snapshot(data2['code'].values.tolist())
        if ret3 == RET_OK:
            res2 = data3[['last_price','option_implied_volatility','option_expiry_date_distance','option_contract_size',
                            'option_delta','option_gamma','option_vega','option_theta','option_rho']]
        else:
            print('error:', data3)
        
        result = result.join(res2)
        return result
    else:
        print('error:', data2)

        
def black_scholes(stock_price, strike_price,option_type,expiry_in_days,volatility,risk_free_return=0.01):
    S=stock_price
    K=strike_price
    r=risk_free_return
    t=expiry_in_days/365
    sigma=volatility/100+1e-6
    
    d1 = (np.log(S/K)+(r+0.5*sigma**2)*t)/(sigma*t**0.5)
    d2 = d1-(sigma*t**0.5)
    
    if option_type=="CALL":
        C = S*norm.cdf(d1)-K*np.exp(-r*t)*norm.cdf(d2)
        return np.round(C,2)
    elif option_type=="PUT":
        P = K*np.exp(-r*t)*norm.cdf(-d2)-S*norm.cdf(-d1)
        return np.round(P,2)
    
def queue_possible_hedge(trd_ctx, options, direction, use_current_price=True, stock_price=0, expiry=30):
    hedge = []
    for idx, opt in options.iterrows():

        if use_current_price:
            price = opt['last_price']
        else:
            price = black_scholes(stock_price,opt['strike_price'],opt['option_type'],expiry,opt['option_implied_volatility'])
        
        ret, acctradinginfo = trd_ctx.acctradinginfo_query(order_type=OrderType.NORMAL, code=opt['code'], price=price)
        if direction=="SHORT":
            qty = -acctradinginfo['max_sell_short'][0]-acctradinginfo['max_position_sell'][0]
        elif direction=="LONG":
            qty = acctradinginfo['max_cash_and_margin_buy'][0]+acctradinginfo['max_buy_back'][0]
        else:
            raise Exception()
        mod = OptionPosition(opt['option_type'], opt['strike_price'], qty, price, opt['option_contract_size'])
        hedge.append(mod)
        time.sleep(3)
    return hedge

def rank_by_min_root(current_position, hedges):
    minroots=[]
    for mod in hedges:
        roots = find_roots(current_position+[mod])
        minroots += [sorted(roots)[0]]
        
    return [x for x, _ in sorted(zip(hedges, minroots),key=lambda i:i[1])], sorted(minroots)

def rank_by_max_root(current_position, hedges):
    maxroots=[]
    for mod in hedges:
        roots = find_roots(current_position+[mod])
        maxroots += [sorted(roots)[-1]]
        
    return [x for x, _ in sorted(zip(hedges, maxroots),key=lambda i:i[1])], sorted(maxroots)


def estimate_integral(options,xstart,xend,intv):
    X = np.arange(xstart,xend,intv)
    Y = []
    for opt in options:
        Y.append(opt.return_function()(X))
    Y = np.array(Y).sum()*intv
    return Y
    
def augment_option(options, num_steps):
    results = []
    for opt in options:
        qty = np.linspace(0,opt.quantity,num_steps).astype(int)
        for q in qty:
            results.append(OptionPosition(opt.option_type, opt.strike_price, q, opt.cost_price, opt.multiplier))
    return results

    
def rank_by_integral(current_position, hedges, xstart, xend, intv):
    intgs=[]
    for mod in hedges:
        intgs += [estimate_integral(current_position+[mod],xstart,xend,intv)]
    return [x for x, _ in sorted(zip(hedges, intgs),key=lambda i:i[1])], sorted(intgs)


def find_maintenance_critical_price(positions,maint_bal,current_stock_price):
    funcs = []
    for idx,row in positions.iterrows():
        p = lambda s:black_scholes(s, row['option_strike_price'],
                      row['option_type'],row['option_expiry_date_distance'],
                      row['option_implied_volatility'],risk_free_return=0.05)
        qty = row['qty']
        nom = row['nominal_price']
        mul = row['option_contract_multiplier']
        fn = lambda s: (nom-p(s))*qty*mul
        funcs += [fn]

    F = lambda p: maint_bal - sum([f(p) for f in funcs])

    r = brentq(F,1,current_stock_price*2)
    return r