import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch.utils.data import DataLoader

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.algos.batch_mppi import BatchMPPI
from torch_mpc.algos.mppi import MPPI
from torch_mpc.cost_functions.waypoint_costmap import WaypointCostMapCostFunction

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.costmappers.linear_costmapper import LinearCostMapper
from maxent_irl_costmaps.utils import get_feature_counts
from maxent_irl_costmaps.networks.mlp import MLP

class MPPIIRL:
    """
    Costmap learner that uses expert data + MPPI optimization to learn costmaps.
    The algorithm is as follows:
    1. Get empirical feature counts across the entire dataset for expert
    2. Iteratively
        a. Sample a batch of data from the expert's dataset
        b. Compute a set of costmaps from the current weight vector.
        c. Use MPPI to optimize a trajectory on the costmap
        d. Get empirical feature counts from the MPPI solver (maybe try the weighted trick)
        e. Match feature expectations
    """
    def __init__(self, expert_dataset, mppi):
        """
        Args:
            expert_dataset: The dataset containing expert demonstrations to imitate
            mppi: The MPPI object to optimize with
        """
        self.expert_dataset = expert_dataset
        self.mppi = mppi
        self.mppi_itrs = 3

        hidden = 128
        self.network = MLP(insize = len(expert_dataset.feature_keys), outsize=hidden, hiddens=[128,])
        self.last_weights = torch.zeros(hidden)

        self.network_opt = torch.optim.Adam(self.network.parameters())
        self.last_lr = 0.001

        self.itr = 0

    def update(self):
        self.itr += 1
        dl = DataLoader(self.expert_dataset, batch_size=1, shuffle=True)
        grads = []
        efc = []
        lfc = []
        deep_features_cache = []

        for i, data in enumerate(dl):
            if i > 64:
                break

            print(i, end='\r')
            map_features = data['map_features'][0]
            map_metadata = {k:v[0] for k,v in data['metadata'].items()}
            expert_traj = data['traj'][0]

            #compute costmap
#            costmap = (map_features * self.weights.view(-1, 1, 1)).sum(dim=0)
#            costmap = self.costmapper.get_costmap(data)[0]

            deep_features = torch.moveaxis(self.network.forward(torch.moveaxis(map_features, 0, -1)), -1, 0)
            costmap = (deep_features * self.last_weights.view(-1, 1, 1)).sum(dim=0)

            #initialize solver
            initial_state = expert_traj[0]
            HACK = {"state":initial_state, "steer_angle":torch.zeros(1)}
            x = self.mppi.model.get_observations(HACK)
            self.mppi.cost_fn.update_costmap(costmap)

            #solve for traj
            for ii in range(self.mppi_itrs):
                self.mppi.get_control(x, step=False)

            #regular version
            traj = self.mppi.last_states

            #weighting version
            trajs = self.mppi.noisy_states.clone()
            weights = self.mppi.last_weights.clone()

            self.mppi.reset()

            #get learner feature counts

            #regular
#            learner_feature_counts = self.expert_dataset.get_feature_counts(traj, map_features, map_metadata)

            #MPPI weight
            learner_feature_counts = get_feature_counts(trajs, deep_features, map_metadata)
            learner_feature_counts = (weights.view(1, -1) * learner_feature_counts).sum(dim=-1)

            expert_feature_counts = get_feature_counts(expert_traj, deep_features, map_metadata)

            lfc.append(learner_feature_counts)
            efc.append(expert_feature_counts)
            grads.append(expert_feature_counts - learner_feature_counts)
            deep_features_cache.append(deep_features)

            """
            #Viz debug
            if (i%10) == 0:
                xmin = map_metadata['origin'][0]
                xmax = xmin + map_metadata['width']
                ymin = map_metadata['origin'][1]
                ymax = ymin + map_metadata['height']
                plt.imshow(costmap.detach(), origin='lower', extent=(xmin, xmax, ymin, ymax))

                plt.plot(traj[:, 0], traj[:, 1], c='r', label='learner')

                plt.plot(expert_traj[:, 0], expert_traj[:, 1], c='b', label='expert')
                plt.title('Itr {}'.format(self.itr))
                plt.legend()

                plt.show()
            """ 

        grads = torch.stack(grads, dim=0)
        lfc = torch.stack(lfc, dim=0)
        efc = torch.stack(efc, dim=0)
        deep_features_cache = torch.stack(deep_features_cache, dim=0)
        deep_features_temp = deep_features_cache.mean(dim=0).mean(dim=-1).mean(dim=-1)

        shallow_grad = grads.mean(dim=0)

        self.network_opt.zero_grad()
        deep_features_temp.backward(gradient=shallow_grad * self.last_weights)
        self.network_opt.step()

        self.last_weights -= shallow_grad.detach() * self.last_lr

        print('__________ITR {}__________'.format(self.itr))
        print('WEIGHTS:\n', self.last_weights.detach().numpy())
        print('LEARNER FC: ', lfc.mean(dim=0).detach())
        print('EXPERT FC:  ', efc.mean(dim=0).detach())

    def visualize(self):
        dl = DataLoader(self.expert_dataset, batch_size=1, shuffle=True)
        with torch.no_grad():
            for i, data in enumerate(dl):
                if i > 20:
                    return

                map_features = data['map_features'][0]
                map_metadata = {k:v[0] for k,v in data['metadata'].items()}
                expert_traj = data['traj'][0]

                #compute costmap
    #            costmap = (map_features * self.weights.view(-1, 1, 1)).sum(dim=0)
    #            costmap = self.costmapper.get_costmap(data)[0]

                deep_features = torch.moveaxis(self.network.forward(torch.moveaxis(map_features, 0, -1)), -1, 0)
                costmap = (deep_features * self.last_weights.view(-1, 1, 1)).sum(dim=0)

                #initialize solver
                initial_state = expert_traj[0]
                HACK = {"state":initial_state, "steer_angle":torch.zeros(1)}
                x = self.mppi.model.get_observations(HACK)
                self.mppi.cost_fn.update_costmap(costmap)

                #solve for traj
                for ii in range(self.mppi_itrs):
                    self.mppi.get_control(x, step=False)

                #regular version
                traj = self.mppi.last_states

                metadata = data['metadata']
                xmin = metadata['origin'][0, 0]
                ymin = metadata['origin'][0, 1]
                xmax = xmin + metadata['width'][0]
                ymax = ymin + metadata['height'][0]
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                
                axs[0].imshow(map_features[1], origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
                m1 = axs[1].imshow(costmap, origin='lower', cmap='coolwarm', extent=(xmin, xmax, ymin, ymax))
                axs[0].plot(expert_traj[:, 0], expert_traj[:, 1], c='y', label='expert')
                axs[1].plot(expert_traj[:, 0], expert_traj[:, 1], c='y')
                axs[0].plot(traj[:, 0], traj[:, 1], c='g', label='learner')
                axs[1].plot(traj[:, 0], traj[:, 1], c='g')

                axs[0].set_title('heightmap high')
                axs[1].set_title('irl cost')
                axs[0].legend()

                plt.colorbar(m1, ax=axs[1])
                plt.show()

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    horizon = 70
    batch_size = 100

    bag_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/rosbags/'
    pp_fp = '/home/yamaha/Desktop/datasets/yamaha_maxent_irl/torch/'

    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp)

    kbm = SteerSetpointKBM(L=3.0, v_target_lim=[3.0, 8.0], steer_lim=[-0.3, 0.3], steer_rate_lim=0.2)

    parameters = {
        'log_K_delta':torch.tensor(10.0)
    }
    kbm.update_parameters(parameters)
    cfn = WaypointCostMapCostFunction(unknown_cost=10., goal_cost=1000., map_params=dataset.metadata)
    mppi = MPPI(model=kbm, cost_fn=cfn, num_samples=2048, num_timesteps=horizon, control_params={'sys_noise':torch.tensor([2.0, 0.5]), 'temperature':0.05})

    mppi_irl = MPPIIRL(dataset, mppi)

    for i in range(100):
        mppi_irl.update()
        with torch.no_grad():
            torch.save({'net':mppi_irl.network, 'weights':mppi_irl.last_weights, 'keys':mppi_irl.expert_dataset.feature_keys}, 'learner_all_data_network/weights_itr_{}.pt'.format(i + 1))

#        if (i % 10) == 0:
            #visualize
    mppi_irl.visualize()
