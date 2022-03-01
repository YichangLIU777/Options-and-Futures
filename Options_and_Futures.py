from cmath import log
import pandas as pd
import numpy as np
import math

# S_0 Current stock price
# K Strike price of the option
# T Time to expiration of the option
# S_T Stock price at expiration date
# r Continuously-compounded risk-free rate
# q Continuously-compounded dividend yield
# C Value of European call option to buy one share of the stock
# P Value of European put option to buy one share of the stock
# D Present value of dividends during the life of the option
# r_u: upper price 比 S_0 增加的比率
# P_u: upper price 
# r_d: down price 比 S_0 减少的比率
# P_d: down price


### Bear Spread: Short put with strike K_1, Long put with strike K_2, where K_1 < K_2
### Bull Spread: Long call with strike K_1, Short Call with strike K_2, where K_1 < K_2

### 代入profit_payoff_call_put(K, S_T, Cost)时，long是+，short是-

def profit_payoff_call_put(K, S_T, Cost):
    ### 计算组合策略时 代入profit_payoff_call_put(K, S_T, Cost)，long是+，short是-
    ### profit_payoff_call_put(K, S_T, Cost)[0]是call profit，profit_payoff_call_put(K, S_T, Cost)[1]是call payoff
    ### profit_payoff_call_put(K, S_T, Cost)[3]是put profit，profit_payoff_call_put(K, S_T, Cost)[4]是put payoff
    Call_Profit = max(S_T-K, 0) - Cost
    Call_Payoff = max(S_T-K, 0)
    Put_Profit = max(K - S_T, 0) - Cost
    Put_Payoff = max(K - S_T, 0)
    return Call_Profit, Call_Payoff, Put_Profit, Put_Payoff

def cost_of_strategy(S_0, StockNumOfShare, CostOfOption, NumOfOption, NumOfShareInContract, Buy_Stock = True, Buy_Option = True):
    if Buy_Stock == True:
        Stock_Cost = S_0*StockNumOfShare
    else:
        Stock_Cost = -S_0*StockNumOfShare
    if Buy_Option == True:
        Option_Cost = CostOfOption*NumOfOption*NumOfShareInContract
    else: 
        Option_Cost = -CostOfOption*NumOfOption*NumOfShareInContract
    return Stock_Cost+Option_Cost

def compounding_PV_FV(PV, FV, compounding_rate, T):
    ### PV\FV\compounding_rate\T, 一般题目里只知道三个，另一个是要求的值，输入None
    ### 例如：compounding_PV_FV(PV, None, compounding_rate, T)
    if PV == None:
        result = FV*(math.e**(-compounding_rate*T))
    if FV == None: 
        result = PV*(math.e**(compounding_rate*T))
    if compounding_rate == None: 
        result = (math.log(FV/PV))/T
    if T == None: 
        result = (math.log(FV/PV))/compounding_rate
    return round(result,4)
    
def conversion_EAR_And_continuous_compounding(EAR, coumpounding_rate, T):
    ### EAR is effective annual rate 
    if coumpounding_rate == None:
        result = math.log(EAR+1)/T
    if EAR == None:
        result = math.e**(coumpounding_rate*T)-1
    if T == None:
        result = math.log(EAR+1)/coumpounding_rate
    return round(result, 4)

def put_call_parity_calculation(S_0, C, K, r, T, P):
    ### put-call parity: S_0 − C = K*e^(−rT) − P
    if S_0 == None:
        result = C + K*math.e**(-r*T) - P
    if C == None:
        result = S_0 - K*math.e**(-r*T) + P
    if P == None:
        result = K*math.e**(-r*T) - S_0 + C
    if K == None:
        result = (math.e**(r*T))*(S_0 - C + P)
    if r == None:
        result = -math.log((S_0 - C + P)/K)/T
    if T == None:
        result = -math.log((S_0 - C + P)/K)/r
    return result

def put_call_parity_strategy(S_0, C, K, r, T, P):
    ### put-call parity: S_0 + P = K*e^(−rT) + C
    if C is not None and P is not None:
        if C + K*math.e**(-r*T) == P + S_0: 
            return "No Arbitrage"
        if C + K*math.e**(-r*T) < P + S_0: 
            PV_stock = K*math.e**(-r*T)
            return "Buy the call, sell the put, sell the stock and deposit " + str(round(PV_stock,4))
        if C + K*math.e**(-r*T) > P + S_0: 
            PV_stock = K*math.e**(-r*T)
            return "Sell the call, buy the put, buy the stock and borrow " + str(round(PV_stock,4))
    if C is None and P is not None:
        PV_stock = K*math.e**(-r*T)
    ### C是none所以要控制P
    ### put-call parity: K*e^(−rT) - S_0 = P - C
        if K*math.e**(-r*T) - S_0 > P:
            return "Buy the put, buy the stock and borrow " + str(round(PV_stock,4))
        elif K*math.e**(-r*T) - S_0 < P:
            return "Sell the put, Sell the stock and deposit/invest " + str(round(PV_stock,4))
        else:
            return "No Arbitrage"
    if P is None and C is not None:
        PV_stock = K*math.e**(-r*T)
        ### put-call parity: S_0 - K*e^(−rT) = C - P
        if S_0 - K*math.e**(-r*T) > C:
            return "Buy the Call, Sell the stock and deposit/invest " + str(round(PV_stock,4))
        elif S_0 - K*math.e**(-r*T) < C:
            return "Sell the Call, Buy the stock and borrow " + str(round(PV_stock,4))
        else:
            return "No Arbitrage"
    if P is None and C is None:
        ### put-call parity: S_0 - K*e^(−rT) = C - P
        theoritical_min_value_for_Call = max(S_0 - K*math.e**(-r*T), 0)
        theoritical_min_value_for_Put = max(K*math.e**(-r*T) - S_0, 0)
        theoritical_max_value_for_CallAndPut = S_0
        return "The range of a call is ("+ str(theoritical_min_value_for_Call) + ", " + str(S_0) + ")" ,\
               "The range of a Put is ("+ str(theoritical_min_value_for_Put) + ", " + str(S_0) + ")"

def risk_neutral(S_0, P_u, P_d, T, r_u, r_d, r, dividend_yield_rate):
    if P_u is None and P_d is None:
        P_u = S_0*(1+r_u); P_d = S_0*(1-r_d)    
    U = P_u/S_0; D = P_d/S_0
    if dividend_yield_rate is None:
        risk_neutral = (math.e**(r*T)-D)/(U-D)
    if dividend_yield_rate is not None:   
        risk_neutral = (math.e**((r-dividend_yield_rate)*T)-D)/(U-D)        
    return risk_neutral

def binomial_call_put_pricing_one_period(S_0, P_u, P_d, T, r, r_u, r_d, risk_neutral, dividend_yield_rate):
    ### P_u\P_d and r_u\r_d 之间有一组是None
    ### 既可以根据P_u, P_d和risk_neutral计算price，也可以直接根据S_0, P_u, P_d计算出risk_neutral做pricing
    if P_u is None and P_d is None:
        P_u = S_0*(1+r_u); P_d = S_0*(1-r_d)
    if risk_neutral is not None:
        Call_Price = math.e**(-r*T)*(risk_neutral*P_u+(1-risk_neutral)*P_d)
        Put_Price = None
    if risk_neutral is None:
        U = P_u/S_0; D = P_d/S_0
        if dividend_yield_rate is None:
            risk_neutral = (math.e**(r*T)-D)/(U-D)
        if dividend_yield_rate is not None:
            risk_neutral = (S_0*math.e**((r - dividend_yield_rate)*T) - P_d)/(P_u - P_d)
        Call_u = max(P_u-S_0, 0); Call_d = max(P_d-S_0, 0)
        Put_u = max(S_0-P_u, 0); Put_d = max(S_0-P_d, 0)
        Call_Price = math.e**(-r*T)*(risk_neutral*Call_u+(1-risk_neutral)*Call_d)
        Put_Price = math.e**(-r*T)*(risk_neutral*Put_u+(1-risk_neutral)*Put_d)
    return Call_Price, Put_Price

def hedge_calculate_NumOfStocks_NumOfPutCall(S_0, r_u, r_d, P_u, P_d, NumOfStock, NumOfPutCall, dividend_yield_rate):
    if P_u is None and P_d is None:
        P_u = S_0*(1+r_u); P_d = S_0*(1-r_d)
    Call_u = max(P_u-S_0, 0); Call_d = max(P_d-S_0, 0)
    call_ratio = (Call_u-Call_d)/(P_u-P_d)
    Put_u = max(S_0-P_u, 0); Put_d = max(S_0-P_d, 0)
    put_ratio = (Put_d-Put_u)/(P_u-P_d)
    if dividend_yield_rate is None:
        if NumOfStock is None:
            return "NumOfStock needed to hegde for selling call: "+str(call_ratio*NumOfPutCall), \
                "NumOfStock needed to hegde for selling put: "+str(put_ratio*NumOfPutCall)
        if NumOfPutCall is None: 
            return "NumOfCall needed to hegde for stocks "+str(NumOfStock/call_ratio), \
                "NumOfPut needed to hegde for stocks "+str(NumOfStock/put_ratio)
    if dividend_yield_rate is not None:
        if NumOfStock is None:
            return "NumOfStock needed to hegde for selling call: "+str(call_ratio*NumOfPutCall*math.e**(-dividend_yield_rate)), \
                "NumOfStock needed to hegde for selling put: "+str(put_ratio*NumOfPutCall*math.e**(-dividend_yield_rate))
        if NumOfPutCall is None: 
            return "NumOfCall needed to hegde for stocks "+str(NumOfStock/(call_ratio*math.e**(-dividend_yield_rate))), \
                "NumOfPut needed to hegde for stocks "+str(NumOfStock/(put_ratio*math.e**(-dividend_yield_rate)))

def binomial_call_put_pricing_two_period(S_0, K, P_u, P_d, T, r, r_u, r_d, dividend_yield_rate):
    if P_u is None and P_d is None:
        P_u = S_0*(1+r_u); P_d = S_0*(1-r_d)
    U = P_u/S_0; D = P_d/S_0
    if dividend_yield_rate is None:
        risk_neutral = (math.e**(r*T)-D)/(U-D)
    if dividend_yield_rate is not None:
        risk_neutral = (math.e**((r-dividend_yield_rate)*T)-D)/(U-D)
    P_uu = S_0*(1+r_u)*(1+r_u); P_ud = S_0*(1+r_u)*(1-r_d); P_dd = S_0*(1-r_d)*(1-r_d)
    Call_uu = max(P_uu-K, 0); Call_ud = max(P_ud-K, 0); Call_dd = max(P_dd-K, 0)
    Call_u = math.e**(-r*T)*(risk_neutral*Call_uu+(1-risk_neutral)*Call_ud)
    Call_d = math.e**(-r*T)*(risk_neutral*Call_ud+(1-risk_neutral)*Call_dd)
    Call_Price = math.e**(-r*T)*(risk_neutral*Call_u+(1-risk_neutral)*Call_d)

    Put_uu = max(K-P_uu, 0); Put_ud = max(K-P_ud, 0); Put_dd = max(K-P_dd, 0)
    Put_u = math.e**(-r*T)*(risk_neutral*Put_uu+(1-risk_neutral)*Put_ud)
    Put_d = math.e**(-r*T)*(risk_neutral*Put_ud+(1-risk_neutral)*Put_dd)
    Put_Price = math.e**(-r*T)*(risk_neutral*Put_u+(1-risk_neutral)*Put_d)
    return "Call Price is " + str(Call_Price), "Put Price is " + str(Put_Price)

def binomial_call_put_pricing_multiple_period_without_dividend(S_0, K, r, T, N, stock_sd):
    delta_t = T/N
    u = math.e**(stock_sd*math.sqrt(delta_t))
    d = 1/u
    S_T = [S_0*(u**(N-i))*(d**(i)) for i in range(N+1)]
    p = (math.e**(r*delta_t) - d) / (u - d)
    risk_neutral = [(math.factorial(N)/(math.factorial(N-i)*math.factorial(i)))*
                            (p**(N-i))*
                            ((1-p)**(i))
                            for i in range(N+1)]
    C_price = (math.e**(-r*T)) * \
            sum( risk_neutral[i]*max((S_T[i]-K),0) for i in range(N+1) )
    P_price = (math.e**(-r*T)) * \
            sum( [risk_neutral[i]*max((K-S_T[i]),0) for i in range(N+1) ])
    return C_price, P_price

def binomial_call_put_pricing_two_period_American_Options(S_0, K, P_u, P_d, T, r, r_u, r_d, dividend_yield_rate):
    if P_u is None and P_d is None:
        P_u = S_0*(1+r_u); P_d = S_0*(1-r_d)
    U = P_u/S_0; D = P_d/S_0
    if dividend_yield_rate is None:
        risk_neutral = (math.e**(r*T)-D)/(U-D)
    if dividend_yield_rate is not None:
        risk_neutral = (math.e**((r-dividend_yield_rate)*T)-D)/(U-D)
    P_uu = S_0*(1+r_u)*(1+r_u); P_ud = S_0*(1+r_u)*(1-r_d); P_dd = S_0*(1-r_d)*(1-r_d)
    Call_uu = max(P_uu-K, 0); Call_ud = max(P_ud-K, 0); Call_dd = max(P_dd-K, 0)
    Call_H_u = math.e**(-r*T)*(risk_neutral*Call_uu+(1-risk_neutral)*Call_ud)
    Call_I_u = max(P_u-K, 0)
    Call_u = max(Call_H_u, Call_I_u)
    Call_H_d = math.e**(-r*T)*(risk_neutral*Call_ud+(1-risk_neutral)*Call_dd)
    Call_I_d = max(P_d-K, 0)
    Call_d = max(Call_H_d, Call_I_d)
    Call_H_Price = math.e**(-r*T)*(risk_neutral*Call_u+(1-risk_neutral)*Call_d)
    Call_I_Price = max(S_0-K, 0)
    Call_Price = max(Call_H_Price, Call_I_Price)

    Put_uu = max(K-P_uu, 0); Put_ud = max(K-P_ud, 0); Put_dd = max(K-P_dd, 0)
    Put_H_u = math.e**(-r*T)*(risk_neutral*Put_uu+(1-risk_neutral)*Put_ud)
    Put_I_u = max(K-P_u, 0)
    Put_u = max(Put_H_u, Put_I_u)
    Put_H_d = math.e**(-r*T)*(risk_neutral*Put_ud+(1-risk_neutral)*Put_dd)
    Put_I_d = max(K-P_d, 0)
    Put_d = max(Put_H_d, Put_I_d)
    Put_H_Price = math.e**(-r*T)*(risk_neutral*Put_u+(1-risk_neutral)*Put_d)
    Put_I_Price = max(K-S_0, 0)
    Put_Price = max(Put_H_Price, Put_I_Price)
    return "Call Price is " + str(Call_Price), "Put Price is " + str(Put_Price)

def forward_current_contract(V, T, r, F, dividend_yield_rate):
    if dividend_yield_rate is not None:
        if F is None:
            return V*math.e**(T*(r-dividend_yield_rate))
        if V is None:
            return F*math.e**(-T*(r-dividend_yield_rate))
    if dividend_yield_rate is  None:
        if F is None:
            return V*math.e**(T*(r))
        if V is None:
            return F*math.e**(-T*(r))





