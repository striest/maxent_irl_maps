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

def get_metrics(experiment, gsv = None, metric_fns = {}, frame_skip=1, vf_downsample=-1, viz=True):
    """
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

            #ensemble
            if hasattr(experiment.network, 'ensemble_forward'):
                #save GPU space in batch (they're copied anyways)
                res = experiment.network.ensemble_forward(map_features[[0]])
                costmap = res['costmap'].mean(dim=1)[0]
                costmap = torch.cat([costmap] * experiment.mppi.B, dim=0)

            #no ensemble
            else:
                res = experiment.network.forward(map_features)
                costmap = res['costmap'][:, 0]

            #set up constraints
            if hasattr(experiment, 'constraint_threshold') and experiment.constraint_threshold > -1e10:
                if 'constraint_logits' in res.keys():
                    constraint_probs = res['constraint_logits'].sigmoid().mean(dim=1)[0]
                    costmap += 1e10 * (constraint_probs > experiment.constraint_threshold).float()
                else:
                    costmap += 1e10 * (costmap > experiment.constraint_threshold).float()


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
            experiment.mppi.cost_fn.data['costmap'] = costmap
            experiment.mppi.cost_fn.data['costmap_metadata'] = map_params

            #set up value function
            if vf_downsample > 0:
                downsample_costmap = torch.nn.functional.max_pool2d(costmap, vf_downsample)
                valuemap_metadata = copy.deepcopy(map_params)
                valuemap_metadata['resolution'] *= vf_downsample
                valuemap = value_iteration(downsample_costmap, valuemap_metadata, goals, gamma=0.95)
                experiment.mppi.cost_fn.data['valuemap'] = valuemap
                experiment.mppi.cost_fn.data['valuemap_metadata'] = valuemap_metadata

            #solve for traj
            for ii in range(experiment.mppi_itrs):
                experiment.mppi.get_control(x, step=False)

            tidx = experiment.mppi.last_cost.argmin()
            traj = experiment.mppi.last_states[tidx].clone()

            trajs = experiment.mppi.noisy_states[tidx].clone()
            weights = experiment.mppi.last_weights[tidx].clone()

            learner_state_visitations = get_state_visitations(trajs, metadata, weights)
            expert_state_visitations = get_state_visitations(expert_traj.unsqueeze(0), metadata)

            """
            #GET GLOBAL STATE VISITATIONS
            gps_x0 = data['gps_traj'][[0]]
            expert_x0 = expert_traj[[0]]

            #calculate the rotation offset to account for frame diff in SO and GPS/
            yaw_offset = quat_to_yaw(expert_x0[:, 3:7]) - quat_to_yaw(gps_x0[:, 3:7])
            poses = torch.stack([
                gps_x0[:, 0],
                gps_x0[:, 1],
                -yaw_offset
            ], axis=-1)

            crop_params = {
                'origin':np.array([-map_params['height'][0].item()/2, -map_params['width'][0].item()/2]),
                'length_x': map_params['height'][0].item(),
                'length_y': map_params['width'][0].item(),
                'resolution': map_params['resolution'][0].item()
            }

            global_state_visitations = gsv.get_state_visitations(poses, crop_params, local=True)[0]
            """
            global_state_visitations = expert_state_visitations

            for k, fn in metric_fns.items():
                metrics_res[k].append(fn(costmap, expert_traj, traj, expert_state_visitations, learner_state_visitations, global_state_visitations).cpu().item())

            xmin = metadata['origin'][0].cpu()
            ymin = metadata['origin'][1].cpu()
            xmax = xmin + metadata['width'].cpu()
            ymax = ymin + metadata['height'].cpu()

            gxmin = gsv.metadata['origin'][0].cpu()
            gymin = gsv.metadata['origin'][1].cpu()
            gxmax = gxmin + gsv.metadata['length_x']
            gymax = gymin + gsv.metadata['length_y']

            for ax in axs:
                ax.cla()

            axs[0].imshow(data['image'].permute(1, 2, 0)[:, :, [2, 1, 0]].cpu())
            axs[0].set_title('FPV')

            axs[1].imshow(costmap[0].cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax), vmax=3.)
            axs[1].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
            axs[1].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g', label='learner')
            axs[1].set_title('costmap')
            axs[1].legend()

            idx = experiment.expert_dataset.feature_keys.index('npts')
            axs[2].imshow(map_features[0, idx].cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
            axs[2].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
            axs[2].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g', label='learner')
            axs[2].set_title('costmap')
            axs[2].legend()

            if vf_downsample > 0:
                axs[2].imshow(valuemap[0].cpu(), origin='lower', cmap='plasma', extent=(xmin, xmax, ymin, ymax))
                axs[2].plot(expert_traj[:, 0].cpu(), expert_traj[:, 1].cpu(), c='y', label='expert')
                axs[2].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='g', label='learner')
                axs[2].set_title('valuemap')
                axs[2].legend()

#            axs[2].imshow(gsv.data.T.cpu(), origin='lower', extent=(gxmin, gxmax, gymin, gymax), vmin=0., vmax=5.)
#            axs[2].scatter(gps_x0[0, 0].cpu(), gps_x0[0, 1].cpu(), color='r', marker='>', s=5.)
#            axs[2].set_title('global visitations')

            axs[-3].imshow(learner_state_visitations.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[-3].set_title('learner SV')

            axs[-2].imshow(expert_state_visitations.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[-2].set_title('expert SV')

            axs[-1].imshow(global_state_visitations.cpu(), origin='lower', extent=(xmin, xmax, ymin, ymax))
            axs[-1].set_title('global SV')

#            for ax in axs[-3:]:
#                ax.scatter(traj[0, 0].cpu(), traj[0, 1].cpu(), c='r', marker='.')

            title = ''
            for k,v in metrics_res.items():
                title += '{}:{:.4f}    '.format(k, v[-1])
            plt.suptitle(title)

            if viz:
#                plt.show()
                plt.pause(1e-2)

#                plt.savefig('res/{:05d}.png'.format(i))

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
