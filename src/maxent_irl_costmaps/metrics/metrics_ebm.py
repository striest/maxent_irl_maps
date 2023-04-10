"""
Collection of metrics for evaluating performance of mexent IRL
"""
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch_mpc.cost_functions.cost_terms.utils import value_iteration

from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer
from maxent_irl_costmaps.networks.baseline_lethal_height import LethalHeightCostmap
from maxent_irl_costmaps.utils import get_state_visitations, quat_to_yaw

def get_metrics_ebm(experiment, gsv = None, metric_fns = {}, frame_skip=1, vf_downsample=-1, viz=True):
    """
    Evaluate the performance of an energy-based model for MPPI trajopt
    Wrapper method that generates metrics for an experiment
    Args:
        experiment: the experiment to compute metrics for
        gsv: global state visitation buffer to use if gps
        metric_fns: A dict of {label:function} (the ones defined in this file) to use to compute metrics
    """
#    plt.show(block=False)
    baseline = LethalHeightCostmap(experiment.expert_dataset).to(experiment.device)

    metrics_res = {k:[] for k in metric_fns.keys()}

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    axs = axs.flatten()
    with torch.no_grad():
        for i in range(0, len(experiment.expert_dataset), frame_skip):
#            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
#            axs = axs.flatten()
            print('{}/{}'.format(i+1, len(experiment.expert_dataset)), end='\r')

            data = experiment.expert_dataset[i]

            #hack back to single dim
            map_features = torch.stack([data['map_features']] * experiment.mppi.B, dim=0)
            metadata = data['metadata']
            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['width'].cpu()
            ymax = ymin + metadata['height'].cpu()
            expert_traj = data['traj']

            #initialize solver
            initial_state = expert_traj[0]
            x0 = {"state":initial_state, "steer_angle":data["steer"][[0]] if "steer" in data.keys() else torch.zeros(1, device=initial_state.device)}
            x = torch.stack([experiment.mppi.model.get_observations(x0)] * experiment.mppi.B, dim=0)

            map_params = {
                'resolution': torch.tensor([metadata['resolution']] * experiment.mppi.B, device=experiment.mppi.device),
                'height': torch.tensor([metadata['height']] * experiment.mppi.B, device=experiment.mppi.device),
                'width': torch.tensor([metadata['width']] * experiment.mppi.B, device=experiment.mppi.device),
                'origin': torch.stack([metadata['origin']] * experiment.mppi.B, dim=0)
            }

            goals = [expert_traj[[-1], :2]] * experiment.mppi.B

            experiment.mppi.reset()
            experiment.mppi.cost_fn.data['goals'] = goals
            experiment.mppi.cost_fn.data['map_features'] = map_features
            experiment.mppi.cost_fn.data['map_metadata'] = map_params

            #solve for traj
            for ii in range(experiment.mppi_itrs):
                experiment.mppi.get_control(x, step=False)

            tidx = experiment.mppi.last_cost.argmin()
            traj = experiment.mppi.last_states[tidx].clone()

            trajs = experiment.mppi.noisy_states[tidx].clone()
            weights = experiment.mppi.last_weights[tidx].clone()

            learner_state_visitations = get_state_visitations(trajs, metadata, weights)
            expert_state_visitations = get_state_visitations(expert_traj.unsqueeze(0), metadata)

            global_state_visitations = expert_state_visitations

            costmap = torch.zeros_like(map_features[:, 0])

            for k, fn in metric_fns.items():
                metrics_res[k].append(fn(costmap, expert_traj, traj, expert_state_visitations, learner_state_visitations, global_state_visitations).cpu().item())

            if viz:
                for ax in axs:
                    ax.cla()

                idx = experiment.expert_dataset.feature_keys.index('height_high')
                #plot the "path integral" of height high
                mppi_res = {
                    'traj': experiment.mppi.last_states.unsqueeze(1),
                    'cmd': experiment.mppi.last_controls.unsqueeze(1),
                    'map_features': map_features,
                    'metadata': map_params
                }
                mppi_feats = experiment.ebm_term.make_training_input(mppi_res).squeeze()

                expert_kbm_traj = {"state": expert_traj, "steer_angle": data["steer"].unsqueeze(-1) if 'steer' in data.keys() else torch.zeros(1, expert_traj.shape[0], device=initial_state.device)}
                expert_kbm_traj = torch.stack([experiment.mppi.model.get_observations(expert_kbm_traj)] * experiment.mppi.B, dim=0)
                expert_cmd = torch.stack([data['cmd']] * experiment.mppi.B, dim=0)
                expert_res = {
                    'traj': expert_kbm_traj.unsqueeze(1),
                    'cmd': expert_cmd.unsqueeze(1),
                    'map_features': map_features,
                    'metadata': map_params
                }
                expert_feats = experiment.ebm_term.make_training_input(expert_res).squeeze()

                rand_res = {
                    'traj': experiment.mppi.noisy_states[:, [0]],
                    'cmd': experiment.mppi.noisy_controls[:, [0]],
                    'map_features': map_features,
                    'metadata': map_params
                }
                rand_feats = experiment.ebm_term.make_training_input(rand_res).squeeze()

                expert_logits = experiment.network.forward(expert_feats.flatten(start_dim=-2))
                learner_logits = experiment.network.forward(mppi_feats.flatten(start_dim=-2))
                rand_logits = experiment.network.forward(rand_feats.flatten(start_dim=-2))
                
                axs[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
                axs[1].imshow(map_features[tidx][idx].cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))

                axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
                axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g', label='learner')
                axs[1].plot(experiment.mppi.noisy_states[tidx, 0, :, 0].cpu(), experiment.mppi.noisy_states[tidx, 0, :, 1].cpu(), c='b', label='rand')

                #plot height high integral
                axs[2].plot(mppi_feats[tidx, :, idx].cpu(), c='g', label='learner height high')
                axs[2].plot(expert_feats[tidx, :, idx].cpu(), c='y', label='expert height high')
                axs[2].plot(rand_feats[tidx, :, idx].cpu(), c='b', label='rand height high')
                axs[2].legend()

                #plot expert speed
                e_speeds = torch.linalg.norm(expert_traj[:, 7:10], axis=-1).cpu()
                l_speeds = traj[:, 3].cpu()
                times = torch.arange(len(e_speeds)) * experiment.mppi.model.dt
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

                title = ''
                for k,v in metrics_res.items():
                    title += '{}:{:.4f}    '.format(k, v[-1])
                plt.suptitle(title)

#                plt.show()
                plt.pause(1e-2)

            #idk why I have to do this
            if i == (len(experiment.expert_dataset)-1):
                break

    plt.close()
    return {k:torch.tensor(v) for k,v in metrics_res.items()}

def expert_cost(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations,
                global_state_visitations
                ):
    return (costmap * expert_state_visitations).sum()

def learner_cost(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations,
                global_state_visitations
                ):
    return (costmap * learner_state_visitations).sum()

def position_distance(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations,
                global_state_visitations
                ):
    return torch.linalg.norm(expert_traj[:, :2] - learner_traj[:, :2], dim=-1).sum()

def kl_divergence(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations,
                global_state_visitations
                ):
    #We want learner onto expert
    #KL(p||q) = sum_p[p(x) * log(p(x)/q(x))]
    return (learner_state_visitations * torch.log((learner_state_visitations + 1e-6) / (expert_state_visitations + 1e-6))).sum()

def kl_divergence_global(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations,
                global_state_visitations
                ):
    #We want learner onto global
    #KL(p||q) = sum_p[p(x) * log(p(x)/q(x))]
    return (learner_state_visitations * torch.log((learner_state_visitations + 1e-6) / (global_state_visitations + 1e-6))).sum()

def modified_hausdorff_distance(
                costmap,
                expert_traj,
                learner_traj,
                expert_state_visitations,
                learner_state_visitations,
                global_state_visitations
                ):
    ap = expert_traj[:, :2]
    bp = learner_traj[:, :2]
    dist_mat = torch.linalg.norm(ap.unsqueeze(0) - bp.unsqueeze(1), dim=-1)
    mhd1 = dist_mat.min(dim=0)[0].mean()
    mhd2 = dist_mat.min(dim=1)[0].mean()
    return max(mhd1, mhd2)

if __name__ == '__main__':
    experiment_fp = '/home/striest/Desktop/experiments/yamaha_maxent_irl/2022-06-29-11-21-25_trail_driving_cnn_deeper_bnorm_exp/itr_50.pt'
    experiment = torch.load(experiment_fp, map_location='cpu')

    gsv_fp = '/home/striest/physics_atv_ws/src/perception/maxent_irl_maps/src/maxent_irl_costmaps/dataset/gsv.pt'
    gsv = torch.load(gsv_fp)

    metrics = {
        'expert_cost':expert_cost,
        'learner_cost':learner_cost,
        'traj':position_distance,
        'kl':kl_divergence,
        'mhd':maidified_hausdorff_distance
    }

    res = get_metrics(experiment, gsv, metrics)
    torch.save(res, 'res.pt')
