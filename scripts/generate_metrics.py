import rosbag
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import argparse
import scipy.spatial
import scipy.interpolate

from torch_mpc.models.steer_setpoint_kbm import SteerSetpointKBM
from torch_mpc.algos.batch_mppi import BatchMPPI

from torch_mpc.cost_functions.generic_cost_function import CostFunction
from torch_mpc.cost_functions.cost_terms.costmap_projection import CostmapProjection
from torch_mpc.cost_functions.cost_terms.valuemap_projection import ValueMapProjection
from torch_mpc.cost_functions.cost_terms.euclidean_distance_to_goal import EuclideanDistanceToGoal

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.dataset.global_state_visitation_buffer import GlobalStateVisitationBuffer
from maxent_irl_costmaps.os_utils import maybe_mkdir
from maxent_irl_costmaps.metrics.metrics import *

from maxent_irl_costmaps.networks.baseline_lethal_height import LethalHeightCostmap

if __name__ == '__main__':
    torch.set_printoptions(sci_mode=False)

    parser = argparse.ArgumentParser()
    parser.add_argument('--save_fp', type=str, required=True, help='path to save figs to')
    parser.add_argument('--model_fp', type=str, required=True, help='Costmap weights file')
    parser.add_argument('--bag_fp', type=str, required=True, help='dir for rosbags to train from')
    parser.add_argument('--preprocess_fp', type=str, required=True, help='dir to save preprocessed data to')
    parser.add_argument('--gsv_buffer_fp', type=str, required=True, help='path to the global state visitation buffer')
    parser.add_argument('--map_topic', type=str, required=False, default='/local_gridmap', help='topic to extract map features from')
    parser.add_argument('--odom_topic', type=str, required=False, default='/integrated_to_init', help='topic to extract odom from')
    parser.add_argument('--image_topic', type=str, required=False, default='/multisense/left/image_rect_color', help='topic to extract images from')
    parser.add_argument('--baseline', action='store_true', required=False, help='set this flag to run baseline map')
    parser.add_argument('--no_costmap', action='store_true', required=False, help='set this flag to run with no costmap i.e. just go to goal')
    parser.add_argument('--constraint', action='store_true', required=False, help='set this flag to run with constraint')
    parser.add_argument('--value_iteration', action='store_true', required=False, help='set this flag to use costmap + value iteration as the MPPI cost function')
    parser.add_argument('--viz', action='store_true', required=False, help='set this flag to visualize output')
    parser.add_argument('--device', type=str, required=False, default='cpu', help='device to run script on')
    args = parser.parse_args()

    model = torch.load(args.model_fp, map_location='cpu').to(args.device)
    model.network.eval()

    if args.constraint:
        model.set_constraint_threshold()

    if args.value_iteration:
        model.mppi.cost_fn = CostFunction([
            (1.0, CostmapProjection()),
            (10.0, ValueMapProjection())
        ]).to(args.device)

    if args.no_costmap:
        model.mppi.cost_fn = CostFunction([
            (10.0, EuclideanDistanceToGoal())
        ]).to(args.device)

    dataset = MaxEntIRLDataset(bag_fp=args.bag_fp, preprocess_fp=args.preprocess_fp, map_features_topic=args.map_topic, odom_topic=args.odom_topic, image_topic=args.image_topic, horizon=model.expert_dataset.horizon, feature_keys=model.expert_dataset.feature_keys).to(args.device)

    model.expert_dataset = dataset

    if args.baseline:
        model.network = LethalHeightCostmap(dataset).to(args.device)

    gsv = torch.load(args.gsv_buffer_fp, map_location='cpu').to(args.device)

    maybe_mkdir(args.save_fp, force=False)
    metrics = {
        'expert_cost':expert_cost,
        'learner_cost':learner_cost,
        'traj':position_distance,
        'kl':kl_divergence,
        'kl_global':kl_divergence_global,
        'mhd': modified_hausdorff_distance
    }

#    for i in range(100):
#        dataset.visualize()
#        plt.show()

    res = get_metrics(model, gsv, metrics, frame_skip=10, vf_downsample=4 if args.value_iteration else -1, viz=args.viz)
#    res = get_metrics(model, gsv, metrics, frame_skip=200, vf_downsample=4 if args.value_iteration else -1, viz=args.viz)
    torch.save(res, os.path.join(args.save_fp, 'metrics.pt'))
