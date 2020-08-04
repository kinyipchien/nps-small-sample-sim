import matplotlib.pyplot as plt
import numpy as np


def coin_flips(n=1, p=0.5, size=10, return_arr=False):
    '''Flips a coin <size> times and prints the sequence and number of Heads'''
    # initialize BitGenerator
    rng = np.random.default_rng()
    
    # flip a coin <n> times with P(Heads)=<p>
    flips = rng.binomial(n=n, p=p, size=size)
    
    # return the array if True
    if return_arr:
        return flips
    # print the sequence of H/T flips and a count of Heads
    print(np.where(flips.astype(bool), 'H', 'T'), f'{flips.sum()} Heads')
    
    
def nps_sim(det_neu_pro, ax, sample_size=22, runs=10000, left_bin=-100, right_bin=100, bins=41, align='left'):
    '''Simulates NPS variability for a given sample_size and returns a histogram and prints some info'''
    # initialize BitGenerator
    rng = np.random.default_rng()
    # sample of Detractors, Neutrals, Promoters
    sample = np.concatenate((-np.ones(det_neu_pro[0])
                         ,np.zeros(det_neu_pro[1])
                         ,np.ones(det_neu_pro[2])))
    
    # bootstrap samples
    bootstrap_samples = rng.choice(sample, size=(runs, sample_size), replace=True)
    # bootstrap NPSs
    bootstrap_nps = bootstrap_samples.mean(axis=1).round(2) * 100
    # bootstrap NPS quantiles
    quantiles = np.sort(bootstrap_nps)
    
    # plot histogram of NPS
    ax.hist(bootstrap_nps, bins=np.linspace(left_bin-0.5, right_bin+0.5, num=bins), align=align)
    # plot settings
    ax.set(title=f'NPS from {sample_size} Customers ({runs} runs)'
          ,xlabel='NPS (rounded to 5s)'
          ,ylabel='Count of runs')
    
    # print some info
    print(f"{'='*79}\n{sample_size} CUSTOMERS")
    print(f"On average, the team's NPS will be {quantiles.mean().astype(int)}")
    print(f'50% of the time, NPS will be between {quantiles[int(runs*0.25)].astype(int)} and {quantiles[int(runs*0.75)].astype(int)}')
    print(f'50% of the time, NPS will be below {quantiles[int(runs*0.25)].astype(int)} and above {quantiles[int(runs*0.75)].astype(int)}')
    print(f'95% of the time, NPS will be between {quantiles[int(runs*0.025)].astype(int)} and {quantiles[int(runs*0.975)].astype(int)}')
    print(f'{(((quantiles < 50).sum() / runs) * 100).astype(int)}% of the time, NPS will be below 50')

    # return array of bootstrap detractors, neutrals, promoters
    return bootstrap_nps

def nps_margin_of_error(det_neu_pro=(1, 4, 17), sample_size=22, plot=False, confidence=0.95, runs=10000):
    '''A calculator that will return a confidence interval for NPS scores based on a 
    tuple containing counts (detractors, neutrals, promoters) and sample_size
    '''
    if len(det_neu_pro) == 1:
        return None, det_neu_pro, None
    # initialize BitGenerator
    rng = np.random.default_rng()
    # sample of Detractors, Neutrals, Promoters to calculate margin of error for
    sample = np.concatenate((-np.ones(det_neu_pro[0])
                         ,np.zeros(det_neu_pro[1])
                         ,np.ones(det_neu_pro[2])))
    # bootstrap samples
    bootstrap_samples = rng.choice(sample, size=(runs, sample_size), replace=True)
    # bootstrap NPSs
    bootstrap_nps = bootstrap_samples.mean(axis=1).round(2) * 100
    # bootstrap NPS quantiles
    quantiles = np.sort(bootstrap_nps)
    # return confidence interval and mean for NPS as (cl, mean, cu)
    return (quantiles[int(runs * (0.5 - confidence/2))].astype(int), (sample.mean() * 100).astype(int), quantiles[int(runs * (0.5 + confidence/2))].astype(int))