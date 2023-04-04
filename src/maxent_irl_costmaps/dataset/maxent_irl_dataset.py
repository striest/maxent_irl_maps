import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import os
import matplotlib.pyplot as plt

from grid_map_msgs.msg import GridMap
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped

from maxent_irl_costmaps.preprocess import load_data
from maxent_irl_costmaps.utils import dict_to
from maxent_irl_costmaps.os_utils import walk_bags

class MaxEntIRLDataset(Dataset):
    """
    Dataset for maxent IRL on costmaps that contains the following:
        1. A set of map features (plus metadata) extracted from rosbags
        2. A trajectory (registered on the map)

    Ok, the ony diff now is that there are multiple bag files and we save the trajdata to a temporary pt file.
    """
    def __init__(self, bag_fp, preprocess_fp, map_features_topic='/local_gridmap', odom_topic='/integrated_to_init', image_topic='/multisense/left/image_rect_color', horizon=70, dt=0.1, fill_value=0, feature_keys=[]):
        """
        Args:
            bag_fp: The bag to get data from
            config_fp: Config file describing model input (we only listen to state)
            map_features_topic: the topic to read map features from
            odom_topic: the topic to read trajectories from
            horizon: The number of trajectory steps to register onto each map
            feature_keys: If default, use whatever features are in the rosbags. Otherwise, use the features in this list

        Note that since the map changes here, we want to treat each as an MDP and make data samples for each map.
        """
        self.bag_fp = bag_fp
        self.preprocess_fp = preprocess_fp
        self.map_features_topic = map_features_topic
        self.odom_topic = odom_topic
        self.image_topic = image_topic
        self.horizon = horizon
        self.dt = dt
        self.fill_value = fill_value #I don't know if this is the best way to do this, but setting the fill value to 0 implies that missing features contribute nothing to the cost.
        self.N = 0
        self.feature_keys = feature_keys
        self.device = 'cpu'

        self.initialize_dataset()

    def initialize_dataset(self):
        """
        Profile the trajectories in the bag to:
            1. Create torch files to reference later
            2. Index each sample of the dataset
            3. Get map feature statistics for normalization
        """
        preprocess = False
        if os.path.exists(self.preprocess_fp):
            x = input('{} already exists. Preprocess again? [y/N]'.format(self.preprocess_fp))
            preprocess = (x.lower() == 'y')
        else:
            os.mkdir(self.preprocess_fp)
            preprocess = True

        print('Preprocess = ', preprocess)

        #save all the data into individual files
        if preprocess:
            for tfp in walk_bags(self.bag_fp):
                raw_fp = os.path.join(self.bag_fp, tfp)
                data, feature_keys = load_data(raw_fp, self.map_features_topic, self.odom_topic, self.image_topic, self.horizon, self.dt, self.fill_value)
                if data is None:
                    continue
                for i in range(len(data['traj'])):
                    pp_fp = os.path.join(self.preprocess_fp, 'traj_{}.pt'.format(self.N))
                    subdata = {k:v[i] for k,v in data.items()}
                    subdata['feature_keys'] = feature_keys

                    #make heights relative.
                    for k in feature_keys:
                        if k in ['height_low', 'height_mean', 'height_high', 'height_max', 'terrain']:
                            idx = feature_keys.index(k)
                            mask = abs(subdata['map_features'][idx] - self.fill_value) < 1e-6
                            subdata['map_features'][idx][~mask] -= subdata['traj'][0, 2]

                    #add kinematic/goal-directed features
                    xmin = subdata['metadata']['origin'][0].item()
                    xmax = xmin + subdata['metadata']['width']
                    ymin = subdata['metadata']['origin'][1].item()
                    ymax = ymin + subdata['metadata']['height']
                    map_xs = torch.linspace(xmin, xmax, subdata['map_features'].shape[1])
                    map_ys = torch.linspace(ymin, ymax, subdata['map_features'].shape[2])
                    map_poses = torch.stack(torch.meshgrid(map_xs, map_ys, indexing='ij'), dim=-1)

                    ego_pos = subdata['traj'][0, :2]
                    goal_pos = subdata['traj'][-1, :2]
                    ego_steer = subdata['steer'][0] if 'steer' in subdata.keys() else 0.
                    ego_speed = subdata['traj'][0, 7].abs().item() #7th elem is linear x

                    ego_dist_x = map_poses[:, :, 0] - ego_pos[0].item()
                    ego_dist_y = map_poses[:, :, 1] - ego_pos[1].item()
                    goal_dist_x = map_poses[:, :, 0] - goal_pos[0].item()
                    goal_dist_y = map_poses[:, :, 1] - goal_pos[1].item()
                    ego_steer_map = torch.ones_like(ego_dist_x) * ego_steer
                    ego_speed_map = torch.ones_like(ego_dist_x) * ego_speed

                    kinematic_features = torch.stack([
                        ego_dist_x,
                        ego_dist_y,
                        goal_dist_x,
                        goal_dist_y,
                        ego_steer_map,
                        ego_speed_map
                    ])

                    subdata['map_features'] = torch.cat([subdata['map_features'], kinematic_features], dim=0)
                    subdata['feature_keys'] = feature_keys + ['ego_dist_x', 'ego_dist_y', 'goal_dist_x', 'goal_dist_y', 'ego_steer', 'ego_speed']

                    torch.save(subdata, pp_fp)
                    self.N += 1
        
        #Actually read all the data to get statistics.
        #need number of trajs, and the mean/std of all the map features.

        if len(self.feature_keys) == 0:
            self.feature_keys = torch.load(os.path.join(self.preprocess_fp, 'traj_0.pt'))['feature_keys']

        #TODO: need some logic for feature key selection
        self.metadata = torch.load(os.path.join(self.preprocess_fp, 'traj_0.pt'))['metadata']
        nfeats = len(self.feature_keys)
        self.feature_mean = torch.zeros(nfeats)
        self.feature_var = torch.zeros(nfeats)

        traj_fps = os.listdir(self.preprocess_fp)
        self.N = len(traj_fps) - 1
        K = 1
        for i, fp in enumerate(traj_fps):
            print('{}/{} ({})'.format(i+1, len(traj_fps), fp), end='\r')
            traj = torch.load(os.path.join(self.preprocess_fp, fp))

            fk_idxs = self.feature_idx_select(traj['feature_keys'])

            mapfeats = traj['map_features'][fk_idxs, 10:-10, 10:-10].reshape(nfeats, -1)
            k = mapfeats.shape[-1]

            mean_new = (self.feature_mean + (mapfeats.sum(dim=-1) / K)) * (K / (K+k))
            x_ssd = (mapfeats - mean_new.view(-1, 1)).pow(2).sum(dim=-1)
            mean_diff = self.feature_mean - mean_new
            x_correction = (mapfeats - mean_new.view(-1, 1)).sum(dim=-1)
            d_ssd = x_ssd + mean_diff * x_correction
            var_new = (self.feature_var + (d_ssd / K)) * (K / (K+k))

            self.feature_mean = mean_new
            self.feature_var = var_new
            K += k

        self.feature_std = self.feature_var.sqrt() + 1e-6
        self.feature_std[~torch.isfinite(self.feature_std)] = 1e-6

    def feature_idx_select(self, feature_inp_keys):
        """
        Get the indexes of feature_inp_keys that index self.feature_keys correctly
        """
        idxs = []
        for fk in self.feature_keys:
            idxs.append(feature_inp_keys.index(fk))
        return idxs

    def visualize(self):
        """
        Get a rough sense of features
        """
        n_panes = len(self.feature_keys) + 1

        nx = int(np.sqrt(n_panes)) + 1
        ny = int(n_panes / nx) + 1

        fig, axs = plt.subplots(nx, ny, figsize=(20, 20))
        axs = axs.flatten()

        data = self[np.random.randint(len(self))]
        traj = data['traj']
        feats = data['map_features']
        metadata = data['metadata']
        xmin = metadata['origin'][0].cpu()
        ymin = metadata['origin'][1].cpu()
        xmax = xmin + metadata['width'].cpu()
        ymax = ymin + metadata['height'].cpu()

        
        for ax, feat, feat_key in zip(axs, feats, self.feature_keys):
            ax.imshow(feat.cpu(), origin='lower', cmap='gray', extent=(xmin, xmax, ymin, ymax))
#            ax.plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='y')
            ax.set_title(feat_key)

        if 'image' in data.keys():
            image =  data['image']
            ax = axs[len(self.feature_keys)]
            ax.imshow(image.permute(1, 2, 0).cpu())
            ax.set_title('Image')

        return fig, axs

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        data_fp = os.path.join(self.preprocess_fp, 'traj_{}.pt'.format(idx))

        data = torch.load(data_fp, map_location=self.device)

        fk_idxs = self.feature_idx_select(data['feature_keys'])

        data['map_features'] = ((data['map_features'][fk_idxs] - self.feature_mean.view(-1, 1, 1)) / self.feature_std.view(-1, 1, 1)).clip(-5, 5)

        data['feature_keys'] = self.feature_keys

        return data

    def to(self, device):
        self.device = device
        self.feature_mean = self.feature_mean.to(self.device)
        self.feature_std = self.feature_std.to(self.device)
        return self

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    bag_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/rosbags_test/'
    pp_fp = '/home/atv/Desktop/datasets/yamaha_maxent_irl/big_gridmaps/torch_test_h75/'

    feature_keys = []
    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp, feature_keys=feature_keys)
    dataset.visualize()

    feature_keys = ['height_high', 'height_low', 'height_mean', 'height_max', 'unknown']
    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp, feature_keys=feature_keys)
    dataset.visualize()
    plt.show()

    feature_keys += ['terrain', 'terrain_slope']
    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp, feature_keys=feature_keys)
    dataset.visualize()
    plt.show()

    feature_keys += ['SVD1', 'SVD2', 'SVD3']
    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp, feature_keys=feature_keys)
    dataset.visualize()
    plt.show()

    feature_keys += ['roughness', 'diff']
    dataset = MaxEntIRLDataset(bag_fp=bag_fp, preprocess_fp=pp_fp, feature_keys=feature_keys)
    dataset.visualize()
    plt.show()
