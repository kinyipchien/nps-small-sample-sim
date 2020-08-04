import matplotlib.pyplot as plt
import numpy as np


def coin_flips(n=1, p=0.5, size=10, return_arr=False):
    '''Flips a coin <size> times and prints the sequence and number of Heads'''
    rng = np.random.default_rng()
    flips = rng.binomial(n=n, p=p, size=size)
    if return_arr:
        return flips
    print(np.where(flips.astype(bool), 'H', 'T'), f'{flips.sum()} Heads')
    
def nps_sim(sample, ax, runs=10000, sample_size=22, left_bin=-100, right_bin=100, bins=41, align='left'):
    '''Simulates NPS variability for a given sample_size and returns a histogram and prints some info'''
    rng = np.random.default_rng()

    simulations = (rng.choice(sample, size=(runs, sample_size)).mean(axis=1).round(2) * 100)

    ax.hist(simulations, bins=np.linspace(left_bin-0.5, right_bin+0.5, num=bins), align=align)
    ax.set(title=f'NPS from {sample_size} Customers ({runs} runs)'
          ,xlabel='NPS (rounded to 5s)'
          ,ylabel='Count of runs')

    quantiles = np.sort(simulations)
    print(f"{'='*79}\n{sample_size} CUSTOMERS")
    print(f"On average, the team's NPS will be {quantiles.mean().astype(int)}")
    print(f'50% of the time, NPS will be between {quantiles[int(runs*0.25)].astype(int)} and {quantiles[int(runs*0.75)].astype(int)}')
    print(f'50% of the time, NPS will be below {quantiles[int(runs*0.25)].astype(int)} and above {quantiles[int(runs*0.75)].astype(int)}')
    print(f'95% of the time, NPS will be between {quantiles[int(runs*0.025)].astype(int)} and {quantiles[int(runs*0.975)].astype(int)}')
    print(f'{(((quantiles < 50).sum() / runs) * 100).astype(int)}% of the time, NPS will be below 50')
    
    return simulations