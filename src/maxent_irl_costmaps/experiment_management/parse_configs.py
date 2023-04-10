"""
Script for converting experiment yamls into the actual objects to run experiments with
"""
import argparse
import yaml
import torch
import matplotlib.pyplot as plt

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM
from torch_mpc.models.skid_steer import SkidSteer

from torch_mpc.algos.batch_mppi import BatchMPPI
from torch_mpc.algos.batch_actlib_mppi import BatchActlibMPPI

from torch_mpc.cost_functions.generic_cost_function import CostFunction
from torch_mpc.cost_functions.cost_terms.euclidean_distance_to_goal import EuclideanDistanceToGoal
from torch_mpc.cost_functions.cost_terms.costmap_projection import CostmapProjection
from torch_mpc.cost_functions.cost_terms.swathed_costmap_projection import SwathedCostmapProjection
from torch_mpc.cost_functions.cost_terms.ebm import EBMCost, ShapedEBMCost

from maxent_irl_costmaps.algos.mppi_irl_speedmaps import MPPIIRLSpeedmaps
from maxent_irl_costmaps.algos.mppi_irl_constraintmaps import MPPIIRLConstraintmaps
from maxent_irl_costmaps.algos.ebm_mppi import EBMMPPI

from maxent_irl_costmaps.networks.mlp import MLP
from maxent_irl_costmaps.networks.resnet import ResnetCostmapCNN, ResnetCostmapSpeedmapCNN, ResnetCostmapSpeedmapCNNEnsemble, ResnetCostmapSpeedmapCNNEnsemble2, LinearCostmapSpeedmapEnsemble2
from maxent_irl_costmaps.networks.constraint_resnet import ResnetCostmapConstraintmapCNNEnsemble
from maxent_irl_costmaps.networks.fourier_resnet import FourierResnetCostmapSpeedmapCNNEnsemble2
from maxent_irl_costmaps.networks.unet import UNet

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

from maxent_irl_costmaps.experiment_management.experiment import Experiment

def setup_experiment(fp):
    """
    Expect the following top-level keys in the YAML:
        1. experiment: high-level params such as where to save to, epochs, etc.
        2. dataset
        3. network
        4. trajopt
        5. cost_function
        6. model
        7. metrics

    Design decision to use case statements instead of dicts of class types in case I want to
    handle params in specific ways for certain classes
    """
    experiment_dict = yaml.safe_load(open(fp, 'r'))
    experiment_keys = [
        'experiment',
        'dataset',
        'network',
        'netopt',
        'trajopt',
        'cost_function',
        'model',
        'metrics'
    ]
    res = {}
    #check validity of experiment YAML
    for k in experiment_keys:
        assert k in experiment_dict.keys(), "Expected key {} in yaml, found these keys: {}".format(k, experiment_dict.keys())

    #move to correct device
    device = experiment_dict['experiment']['device'] if 'device' in experiment_dict['experiment'].keys() else 'cpu'

    res['params'] = experiment_dict

    #setup dataset
    dataset_params = experiment_dict['dataset']
    if dataset_params['type'] == 'MaxEntIRLDataset':
        res['dataset'] = MaxEntIRLDataset(**dataset_params['params']).to(device)
    else:
        print('Unsupported dataset type {}'.format(dataset_params['type']))
        exit(1)

    #setup network
    network_params = experiment_dict['network']
    if network_params['type'] == 'ResnetCostmapCNN':
        res['network'] = ResnetCostmapCNN(
            in_channels = len(res['dataset'].feature_keys),
            **network_params['params']
        ).to(device)

    elif network_params['type'] == 'ResnetCostmapSpeedmapCNN':
        res['network'] = ResnetCostmapSpeedmapCNN(
            in_channels = len(res['dataset'].feature_keys),
            **network_params['params']
        ).to(device)

    elif network_params['type'] == 'ResnetCostmapSpeedmapCNNEnsemble':
        res['network'] = ResnetCostmapSpeedmapCNNEnsemble(
            in_channels = len(res['dataset'].feature_keys),
            **network_params['params']
        ).to(device)

    elif network_params['type'] == 'ResnetCostmapSpeedmapCNNEnsemble2':
        res['network'] = ResnetCostmapSpeedmapCNNEnsemble2(
            in_channels = len(res['dataset'].feature_keys),
            **network_params['params']
        ).to(device)

    elif network_params['type'] == 'LinearCostmapSpeedmapEnsemble2':
        res['network'] = LinearCostmapSpeedmapEnsemble2(
            in_channels = len(res['dataset'].feature_keys),
            **network_params['params']
        ).to(device)

    elif network_params['type'] == 'ResnetCostmapConstraintmapCNNEnsemble':
        res['network'] = ResnetCostmapConstraintmapCNNEnsemble(
            in_channels = len(res['dataset'].feature_keys),
            **network_params['params']
        ).to(device)

    elif network_params['type'] == 'FourierResnetCostmapSpeedmapCNNEnsemble2':
        res['network'] = FourierResnetCostmapSpeedmapCNNEnsemble2(
            in_channels = len(res['dataset'].feature_keys),
            **network_params['params']
        ).to(device)

    elif network_params['type'] == 'UNet':
        channels = len(res['dataset'].feature_keys)
        nx = int(res['dataset'].metadata['width'] / res['dataset'].metadata['resolution'])
        ny = int(res['dataset'].metadata['height'] / res['dataset'].metadata['resolution'])
        res['network'] = UNet(
            insize = [channels, nx, ny],
            outsize = [1, nx, ny],
            **network_params['params']
        )

    elif network_params['type'] == 'MLP':
        res['network'] = MLP(**network_params['params'])
    else:
        print('Unsupported network type {}'.format(network_params['type']))
        exit(1)

    #setup network opt
    netopt_params = experiment_dict['netopt']
    if netopt_params['type'] == 'Adam':
        res['netopt'] = torch.optim.Adam(res['network'].parameters(), **netopt_params['params'])
    elif netopt_params['type'] == 'AdamW':
        res['netopt'] = torch.optim.AdamW(res['network'].parameters(), **netopt_params['params'])
    else:
        print('Unsupported netopt type {}'.format(netopt_params['type']))
        exit(1)

    #setup model
    model_params = experiment_dict['model']
    if model_params['type'] == 'SteerSetpointKBM':
        res['model'] = SteerSetpointKBM(**model_params['params']).to(device)
    elif model_params['type'] == 'SteerSetpointThrottleKBM':
        res['model'] = SteerSetpointThrottleKBM(**model_params['params']).to(device)
    elif model_params['type'] == 'SkidSteer':
        res['model'] = SkidSteer(**model_params['params']).to(device)
    else:
        print('Unsupported model type {}'.format(model_params['type']))
        exit(1)

    #setup cost function
    cost_function_params = experiment_dict['cost_function']
    terms = []
    for term in cost_function_params['terms']:
        params = term['params'] if term['params'] is not None else {}

        if term['type'] == 'EuclideanDistanceToGoal':
            terms.append((
                term['weight'],
                EuclideanDistanceToGoal(**params)
            ))
        elif term['type'] == 'CostmapProjection':
            terms.append((
                term['weight'],
                CostmapProjection(**params)
            ))
        elif term['type'] == 'SwathedCostmapProjection':
            terms.append((
                term['weight'],
                SwathedCostmapProjection(**params)
            ))
        elif term['type'] == 'EBMCost':
            terms.append((
                term['weight'],
                EBMCost(ebm=res['network'], **params)
            ))
        elif term['type'] == 'ShapedEBMCost':
            terms.append((
                term['weight'],
                ShapedEBMCost(ebm=res['network'], **params)
            ))
        else:
            print('Unsupported cost term type {}'.format(term['type']))
            exit(1)
    
    res['cost_function'] = CostFunction(
        terms
    ).to(device)

    #setup trajopt
    trajopt_params = experiment_dict['trajopt']
    if trajopt_params['type'] == 'BatchMPPI':
        res['trajopt'] = BatchMPPI(
            model = res['model'],
            cost_fn = res['cost_function'],
            num_timesteps = res['dataset'].horizon,
            batch_size = experiment_dict['algo']['params']['batch_size'],
            **trajopt_params['params']
        ).to(device)
    elif trajopt_params['type'] == 'BatchActlibMPPI':
        actlib = torch.load(trajopt_params['actlib_path'])
        actlib = actlib[:trajopt_params['actlib_n']]
        res['trajopt'] = BatchActlibMPPI(
            model = res['model'],
            cost_fn = res['cost_function'],
            actlib=actlib,
            num_timesteps = res['dataset'].horizon,
            batch_size = experiment_dict['algo']['params']['batch_size'],
            **trajopt_params['params']
        ).to(device)
    else:
        print('Unsupported trajopt function type {}'.format(trajopt_params['type']))
        exit(1)

    #setup algo
    algo_params = experiment_dict['algo']
    if algo_params['type'] == 'MPPIIRL':
        res['algo'] = MPPIIRL(
            network = res['network'],
            opt = res['netopt'],
            expert_dataset = res['dataset'],
            mppi = res['trajopt'],
            **algo_params['params']
        ).to(device)

    elif algo_params['type'] == 'MPPIIRLSpeedmaps':
        res['algo'] = MPPIIRLSpeedmaps(
            network = res['network'],
            opt = res['netopt'],
            expert_dataset = res['dataset'],
            mppi = res['trajopt'],
            **algo_params['params']
        ).to(device)

    elif algo_params['type'] == 'MPPIIRLConstraintmaps':
        res['algo'] = MPPIIRLConstraintmaps(
            network = res['network'],
            opt = res['netopt'],
            expert_dataset = res['dataset'],
            mppi = res['trajopt'],
            **algo_params['params']
        ).to(device)
    elif algo_params['type'] == 'EBMMPPI':
        res['algo'] = EBMMPPI(
            network = res['network'],
            opt = res['netopt'],
            expert_dataset = res['dataset'],
            mppi = res['trajopt'],
            **algo_params['params']
        ).to(device)
    else:
        print('Unsupported algo type {}'.format(algo_params['type']))
        exit(1)

    #setup experiment
    experiment_params = experiment_dict['experiment']
    res['experiment'] = Experiment(
        algo = res['algo'],
        params = res['params'],
        **experiment_params
    ).to(device)

    return res

#TEST
if __name__ == '__main__':
    fp = '../../../configs/training/test.yaml'
    res = setup_experiment(fp)

    print({k:v.shape if isinstance(v, torch.Tensor) else v for k,v in res['dataset'][1].items()})

#    for i in range(10):
#        res['dataset'].visualize()
#        plt.show()

    res['experiment'].run()
