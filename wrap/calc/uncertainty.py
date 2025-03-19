'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.
 
Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.
 
                 PACIFIC NORTHWEST NATIONAL LABORATORY
                              operated by
                                BATTELLE
                                for the
                   UNITED STATES DEPARTMENT OF ENERGY
                    under Contract DE-AC05-76RL01830
'''

import scipy.stats as stats


def ensemble_margin(ensemble_std, n_ensembles):
    """Calculate the 95% margin of error from the standard deviation of ensemble predictions 
       using Student's t-distribution

    Args:
       ensemble_std (torch.tensor or numpy.array): Standard deviations of individual predictions.
       n_ensembles (int): Number of models in the ensemble.
    """
    # Calculate the t-value for a 95% confidence interval
    t_value = stats.t.ppf(0.975, n_ensembles - 1) 

    # Calculate margine of error
    return t_value * ensemble_std / (n_ensembles ** 0.5)

