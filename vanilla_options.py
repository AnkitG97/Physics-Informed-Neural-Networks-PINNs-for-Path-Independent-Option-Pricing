import numpy as np
from scipy.stats import norm
import tensorflow as tf


def european_true_solution(apath, x_0, r, q, sigma, T, steps, Strike, t, dt):
    '''
    Given a path, it outputs the true solution for a vanilla European call option.
    '''
    true_solution = []

    for k in range(steps+1):
        tau = T - t[k]  # time to maturity
        if tau == 0:
            # At maturity, option payoff is max(S-K,0)
            payoff = max(apath[k] - Strike, 0)
            true_solution.append(payoff)
        else:
            d1 = (np.log(apath[k]/Strike) + (r - q + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
            d2 = d1 - sigma * np.sqrt(tau)
            call_price = apath[k] * np.exp(-q * tau) * norm.cdf(d1) - Strike * np.exp(-r * tau) * norm.cdf(d2)
            true_solution.append(call_price)
    
    return true_solution

def european_terminal_condition(apath, Strike):
    '''
    Given a path, it outputs the terminal condition g(Y_T) for a European call option.
    '''

    return np.maximum(apath[-1] - Strike, 0)

def european_geometric_payoff(path, Strike):
    '''
    Given a path, it outputs the geometric payoff for a European option.
    '''
    final_price = path[:, -1:]
    return tf.maximum(final_price - Strike, 0.0)