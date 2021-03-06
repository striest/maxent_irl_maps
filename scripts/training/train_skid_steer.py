import torch
import os
import argparse
import matplotlib.pyplot as plt

from torch_mpc.models.skid_steer import SkidSteer
from torch_mpc.algos.mppi import MPPI
from torch_mpc.cost_functions.waypoint_costmap import WaypointCostMapCostFunction

from maxent_irl_costmaps.dataset.maxent_irl_dataset import MaxEntIRLDataset
from maxent_irl_costmaps.algos.mppi_irl import MPPIIRL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_fp', type=str, required=True, help='dir to save experiment results')
    parser.add_argument('--rosbag_dir', type=str, required=True, help='dir for rosbags to train from')
    parser.add_argument('--preprocess_dir', type=str, required=True, help='dir to save preprocessed data to')
    parser.add_argument('--map_topic', type=str, required=False, default='/local_gridmap', help='topic to extract map features from')
    parser.add_argument('--odom_topic', type=str, required=False, default='/warty/odom', help='topic to extract odom from')
    parser.add_argument('--image_topic', type=str, required=False, default='/warty/stereo_left/image_rect_color/compressed', help='topic to extract image from')
    parser.add_argument('--epochs', type=int, required=False, default=10, help='number of epochs to run')
    parser.add_argument('--batch_size', type=int, required=False, default=64, help='batch size')
    parser.add_argument('--horizon', type=int, required=False, default=70, help='number of mppi steps to optimize over')
    parser.add_argument('--model_fp', type=str, required=False, default=None, help='fp to old experiment if testing/fine tuning')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    dataset = MaxEntIRLDataset(bag_fp=args.rosbag_dir, preprocess_fp=args.preprocess_dir, map_features_topic=args.map_topic, odom_topic=args.odom_topic, image_topic=args.image_topic, horizon=int(args.horizon) * 1.0)

    for i in range(10):
        dataset.visualize()
        plt.show()

    vmax = 2.0
    wmax = 1.5
    model = SkidSteer(v_lim=[-vmax, vmax], w_lim=[-wmax, wmax])
    
    cfn = WaypointCostMapCostFunction(unknown_cost=0., goal_cost=10.0, map_params=dataset.metadata)
    mppi  = MPPI(model=model, cost_fn=cfn, num_samples=2048, num_timesteps=args.horizon, control_params={'sys_noise':torch.tensor([0.2 * vmax, 0.2 * wmax]), 'temperature':0.2})

    if args.model_fp:
        mppi_irl = torch.load(args.model_fp)
    else:
        mppi_irl = MPPIIRL(dataset, mppi, args.batch_size)
        mppi_irl.mppi_itrs = 20

    mppi_irl.visualize()

    for ei in range(args.epochs):
        mppi_irl.update()

    torch.save(mppi_irl, 'baseline.pt')
#        if ((ei+1) % 5) == 0:
#            mppi_irl.visualize()
    mppi_irl.visualize()
