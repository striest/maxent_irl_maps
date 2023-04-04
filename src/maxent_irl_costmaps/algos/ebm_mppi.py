import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import tqdm

from torch.utils.data import DataLoader

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset

class EBMMPPI:
    """
    Train an EBM to predict expert trajectories using MPPI
    """
    def __init__(self, network, opt, expert_dataset, mppi, mppi_itrs=10, batch_size=12, device='cpu'):
        """
        Args:
            network: the network to use as the EBM
            opt: The optimizer for the network
            expert_dataset: the dataset whose expert demonstrations we're trying to imitate
            mppi: The MPPI optimizer
        """
        self.expert_dataset = expert_dataset
        self.mppi = mppi
        self.mppi_itrs = mppi_itrs

        self.network = network
        self.ebm_term = [x for x in self.mppi.cost_fn.cost_terms if str(x) in ['EBM', 'Shaped EBM']]
        assert len(self.ebm_term) == 1, 'expected 1 EBM term in cfn, got {}'.format(len(self.ebm_term))
        self.ebm_term = self.ebm_term[0]
        assert id(self.ebm_term.ebm) == id(self.network), 'ebm model and newtwork not same object'

        print(self.network)
        print(sum([x.numel() for x in self.network.parameters()]))
        print(expert_dataset.feature_keys)
        self.network_opt = opt

        self.batch_size = batch_size

        self.itr = 0
        self.device = device

    def update(self, n=-1):
        self.itr += 1
        dl = DataLoader(self.expert_dataset, batch_size=self.batch_size, shuffle=True)
        for i, batch in enumerate(dl):
            if n > -1 and i >= n:
                break

            #skip the last batch in the dataset as MPPI batching forces a fixed size
            if batch['traj'].shape[0] < self.batch_size:
                break

            print('{}/{}'.format(i+1, int(len(self.expert_dataset)/self.batch_size)), end='\r')
            self.gradient_step(batch)

        print('_____ITR {}_____'.format(self.itr))

    def gradient_step(self, batch):
        assert batch['metadata']['resolution'].std() < 1e-4, "got mutliple resolutions in a batch, which we currently don't support"

        #initialize metadata for cost function
        map_params = batch['metadata']
        map_features = batch['map_features']
        #initialize goals for cost function
        expert_traj = batch['traj']
        goals = [traj[[-1], :2] for traj in expert_traj]

        #initialize initial state for MPPI
        initial_states = expert_traj[:, 0]
        x0 = {
            'state': initial_states,
            'steer_angle': batch['steer'][:, [0]] if 'steer' in batch.keys() else torch.zeros(initial_states.shape[0], device=initial_states.device)
        }
        x = self.mppi.model.get_observations(x0)

        #set up the solver
        self.mppi.reset()
        self.mppi.cost_fn.data['goals'] = goals
        self.mppi.cost_fn.data['map_features'] = map_features
        self.mppi.cost_fn.data['map_metadata'] = map_params

        #run MPPI
        for ii in range(self.mppi_itrs):
            with torch.no_grad():
                self.mppi.get_control(x, step=False)

        #weighting version
        trajs = self.mppi.noisy_states.clone()
        weights = self.mppi.last_weights.clone()

        learner_res = {
            'traj': self.mppi.noisy_states,
            'cmd': self.mppi.noisy_controls,
            'map_features': map_features,
            'metadata': map_params
        }
        learner_feats = self.ebm_term.make_training_input(learner_res)

        expert_kbm_traj = {"state": expert_traj, "steer_angle": batch["steer"].unsqueeze(-1) if 'steer' in batch.keys() else torch.zeros(1, expert_traj.shape[0], device=initial_state.device)}
        expert_kbm_traj = self.mppi.model.get_observations(expert_kbm_traj)
        expert_res = {
            'traj': expert_kbm_traj.unsqueeze(1),
            'cmd': batch['cmd'].unsqueeze(1),
            'map_features': map_features,
            'metadata': map_params
        }
        expert_feats = self.ebm_term.make_training_input(expert_res)

        expert_logits = self.network.forward(expert_feats.flatten(start_dim=-2))
        learner_logits = self.network.forward(learner_feats.flatten(start_dim=-2))
        objective = (expert_logits - 1.).pow(2) + (learner_logits - 0.).pow(2)
        loss = objective.mean()

#        fig, axs = plt.subplots(1, 1, figsize=(12, 12))
#        for i in range(len(expert_logits)):
#            axs.plot(expert_logits[i, 0].detach().cpu(), c='y', alpha=0.5)
#            iii = torch.randint(learner_logits.shape[1], (1,)).squeeze()
#            axs.plot(learner_logits[i, iii].detach().cpu(), c='r', alpha=0.5)
#            iii = learner_logits[i].sum(dim=-1).argmax()
#            axs.plot(learner_logits[i, iii].detach().cpu(), c='b', alpha=0.5)
#        plt.show()

        self.network_opt.zero_grad()
        loss.backward()
        self.network_opt.step()

        avg_expert_scores = expert_logits.mean().detach().cpu()
        avg_learner_scores = learner_logits.mean().detach().cpu()
        top_1p_learner_scores = torch.quantile(learner_logits, 0.99).detach()
        avg_diff = avg_expert_scores - avg_learner_scores
        avg_top_learner_score = learner_logits.mean(dim=-1).max(dim=1)[0].mean()

        print('____________________________________________________________')
        print('Avg. expert scores:      {:.4f}'.format(avg_expert_scores.item()))
        print('Avg. learner scores:     {:.4f}'.format(avg_learner_scores.item()))
        print('top 1%. learner scores:  {:.4f}'.format(top_1p_learner_scores.item()))
        print('Avg. top learner scores: {:.4f}'.format(avg_top_learner_score.item()))
        print('Avg. Disc. real - fake:  {:.4f}'.format(avg_diff.item()))
        
        self.mppi.reset()

    def visualize(self, idx=-1):
        if idx == -1:
            idx = np.random.randint(len(self.expert_dataset))

        with torch.no_grad():
            data = self.expert_dataset[idx]

            #hack back to single dim
            map_features = torch.stack([data['map_features']] * self.mppi.B, dim=0)
            metadata = data['metadata']
            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['width'].cpu()
            ymax = ymin + metadata['height'].cpu()
            expert_traj = data['traj']

            #initialize solver
            initial_state = expert_traj[0]
            x0 = {"state":initial_state, "steer_angle":data["steer"][[0]] if 'steer' in data.keys() else torch.zeros(1, device=initial_state.device)}
            x = torch.stack([self.mppi.model.get_observations(x0)] * self.mppi.B, dim=0)

            map_params = {
                'resolution': torch.tensor([metadata['resolution']] * self.mppi.B, device=self.mppi.device),
                'height': torch.tensor([metadata['height']] * self.mppi.B, device=self.mppi.device),
                'width': torch.tensor([metadata['width']] * self.mppi.B, device=self.mppi.device),
                'origin': torch.stack([metadata['origin']] * self.mppi.B, dim=0)
            }

            goals = [expert_traj[[-1], :2]] * self.mppi.B

            self.mppi.reset()
            self.mppi.cost_fn.data['goals'] = goals
            self.mppi.cost_fn.data['map_features'] = map_features
            self.mppi.cost_fn.data['map_metadata'] = map_params

            #solve for traj
            for ii in range(self.mppi_itrs):
                self.mppi.get_control(x, step=False)

            tidx = self.mppi.last_cost.argmin()
            traj = self.mppi.last_states[tidx]

            metadata = data['metadata']
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
            axs = axs.flatten()
            
            idx = self.expert_dataset.feature_keys.index('height_high')
            #plot the "path integral" of height high
            mppi_res = {
                'traj': self.mppi.last_states.unsqueeze(1),
                'cmd': self.mppi.last_controls.unsqueeze(1),
                'map_features': map_features,
                'metadata': map_params
            }
            mppi_feats = self.ebm_term.make_training_input(mppi_res).squeeze()

            expert_kbm_traj = {"state": expert_traj, "steer_angle": data["steer"].unsqueeze(-1) if 'steer' in data.keys() else torch.zeros(1, expert_traj.shape[0], device=initial_state.device)}
            expert_kbm_traj = torch.stack([self.mppi.model.get_observations(expert_kbm_traj)] * self.mppi.B, dim=0)
            expert_cmd = torch.stack([data['cmd']] * self.mppi.B, dim=0)
            expert_res = {
                'traj': expert_kbm_traj.unsqueeze(1),
                'cmd': expert_cmd.unsqueeze(1),
                'map_features': map_features,
                'metadata': map_params
            }
            expert_feats = self.ebm_term.make_training_input(expert_res).squeeze()

            rand_res = {
                'traj': self.mppi.noisy_states[:, [0]],
                'cmd': self.mppi.noisy_controls[:, [0]],
                'map_features': map_features,
                'metadata': map_params
            }
            rand_feats = self.ebm_term.make_training_input(rand_res).squeeze()

            expert_logits = self.network.forward(expert_feats.flatten(start_dim=-2))
            learner_logits = self.network.forward(mppi_feats.flatten(start_dim=-2))
            rand_logits = self.network.forward(rand_feats.flatten(start_dim=-2))
            
            axs[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            axs[1].imshow(map_features[tidx][idx].cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))

            axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
            axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g', label='learner')
            axs[1].plot(self.mppi.noisy_states[tidx, 0, :, 0].cpu(), self.mppi.noisy_states[tidx, 0, :, 1].cpu(), c='b', label='rand')

            #plot height high integral
            axs[2].plot(mppi_feats[tidx, :, idx].cpu(), c='g', label='learner height high')
            axs[2].plot(expert_feats[tidx, :, idx].cpu(), c='y', label='expert height high')
            axs[2].plot(rand_feats[tidx, :, idx].cpu(), c='b', label='rand height high')
            axs[2].legend()

            #plot expert speed
            e_speeds = torch.linalg.norm(expert_traj[:, 7:10], axis=-1).cpu()
            l_speeds = traj[:, 3].cpu()
            times = torch.arange(len(e_speeds)) * self.mppi.model.dt
            axs[3].plot(times, e_speeds, label='expert speed', c='y')
            axs[3].plot(times, l_speeds, label='learner speed', c='g')

            #plto ebm costs
            axs[4].plot(expert_logits[tidx].cpu(), c='y', label='expert energy')
            axs[4].plot(learner_logits[tidx].cpu(), c='g', label='learner energy')
            axs[4].plot(rand_logits[tidx].cpu(), c='b', label='rand energy')
            axs[4].legend()

            axs[0].set_title('FPV')
            axs[1].set_title('heightmap high')
            axs[2].set_title('height high path integral')
            axs[3].set_title('speed')
            axs[4].set_title('Energy fn')

            for i in [1, 2]:
                axs[i].set_xlabel('X(m)')
                axs[i].set_ylabel('Y(m)')

            axs[3].set_xlabel('T(s)')
            axs[3].set_ylabel('Speed (m/s)')
            axs[3].legend()

            axs[1].legend()

        return fig, axs

    def to(self, device):
        self.device = device
        self.expert_dataset = self.expert_dataset.to(device)
        self.mppi = self.mppi.to(device)
        self.network = self.network.to(device)
        return self
