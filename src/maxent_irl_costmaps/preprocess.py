import rosbag
import torch
import numpy as np
import scipy.interpolate, scipy.spatial
import cv2
from cv_bridge import CvBridge

from maxent_irl_costmaps.geometry_utils import TrajectoryInterpolator

def load_traj(bag_fp, odom_topic, dt):
    traj = []
    vels = []
    timestamps = []

    bag = rosbag.Bag(bag_fp, 'r')
    for topic, msg, t in bag.read_messages(topics=[odom_topic]):
        if topic == odom_topic:
            pose = msg.pose.pose
            twist = msg.twist.twist
            p = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ])

            if len(timestamps) == 0 or (msg.header.stamp.to_sec() - timestamps[-1] > 1e-6):
                traj.append(p)
                timestamps.append(msg.header.stamp.to_sec())

    traj = np.stack(traj, axis=0)
    timestamps = np.array(timestamps)

    traj_interp = TrajectoryInterpolator(timestamps, traj)

    start_time = timestamps[0]
    targets = np.arange(timestamps[0], timestamps[-1], dt)

    traj = traj_interp(targets)

    return traj

def load_data(bag_fp, map_features_topic, odom_topic, image_topic, horizon, dt, fill_value, steer_angle_topic='/ros_talon/current_position', gps_topic='/odometry/filtered_odom', cmd_topic='/cmd'):
    """
    Extract map features and trajectory data from the bag.
    """
    print(bag_fp)
    map_features_list = []
    traj = []
    timestamps = []
    dataset = []

    #steer angle needed for yamaha
    steer_angles = []
    steer_timestamps = []

    #gps needed for global state visitations
    gps_poses = []
    gps_timestamps = []

    #cmd needed for models
    cmds = []
    cmd_timestamps = []

    bag = rosbag.Bag(bag_fp, 'r')
    for topic, msg, t in bag.read_messages(topics=[map_features_topic, odom_topic, steer_angle_topic, gps_topic, cmd_topic]):
        if topic == odom_topic:
            pose = msg.pose.pose
            twist = msg.twist.twist
            p = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ])

#            if len(timestamps) == 0 or (msg.header.stamp.to_sec() - timestamps[-1] > 1e-6):
            if len(timestamps) == 0 or (t.to_sec() - timestamps[-1] > 1e-6):
                traj.append(p)
#                timestamps.append(msg.header.stamp.to_sec())
                timestamps.append(t.to_sec())

        ## TODO: handle maps in their own loop ## 
#        if topic == map_features_topic:
#            map_features_list.append(msg)

        if topic == steer_angle_topic:
            steer_angles.append(msg.data)
            steer_timestamps.append(t.to_sec())

        if topic == gps_topic:
            pose = msg.pose.pose
            twist = msg.twist.twist
            gps_state = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
                twist.linear.x,
                twist.linear.y,
                twist.linear.z,
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ])
#            if len(gps_timestamps) == 0 or (msg.header.stamp.to_sec() - gps_timestamps[-1] > 1e-6):
            if len(gps_timestamps) == 0 or (t.to_sec() - gps_timestamps[-1] > 1e-6):
                gps_poses.append(gps_state)
#                gps_timestamps.append(msg.header.stamp.to_sec())
                gps_timestamps.append(t.to_sec())

        if topic == cmd_topic:
            cmd = np.array([
                msg.twist.linear.x,
                msg.twist.angular.z
            ])
            cmds.append(cmd)
#            cmd_timestamps.append(msg.header.stamp.to_sec())
            cmd_timestamps.append(t.to_sec())

    traj = np.stack(traj, axis=0)
    timestamps = np.array(timestamps)

    #edge case check
    idxs = np.argsort(timestamps)

    #filter out maps that dont have enough trajectory
#    map_features_list = [x for x in map_features_list if (x.info.header.stamp.to_sec() > timestamps.min()) and (x.info.header.stamp.to_sec() < timestamps.max() - (horizon*dt))]

#    if len(map_features_list) == 0:
#        return None, None

    traj_interp = TrajectoryInterpolator(timestamps, traj)

    if len(steer_angles) > 0:
        steer_angles = np.stack(steer_angles)
        interp_steer = scipy.interpolate.interp1d(steer_timestamps, steer_angles, fill_value="extrapolate") #this is a bit sketchy

    if len(gps_poses) > 0:
        gps_traj = np.stack(gps_poses, axis=0)
        gps_timestamps = np.array(gps_timestamps)
        gps_interp = TrajectoryInterpolator(gps_timestamps, gps_traj)

    if len(cmds) > 0:
        cmds = np.stack(cmds, axis=0)
        cmd_timestamps = np.array(cmd_timestamps)
        cmd_throttle_interp = scipy.interpolate.interp1d(cmd_timestamps, cmds[:, 0], fill_value='extrapolate')
        cmd_steer_interp = scipy.interpolate.interp1d(cmd_timestamps, cmds[:, 1], fill_value='extrapolate')

    map_target_times = []

    ## TODO: make this loop through the bag (add a break witht he above filter) ##
    #get a registered trajectory for each map.
    i = 0
    for topic, map_features, t in bag.read_messages(topics=[map_features_topic]):
        print('{}'.format(i+1), end='\r')
        map_feature_data = []
        map_feature_keys = []
        mf_nx = int(map_features.info.length_x/map_features.info.resolution)
        mf_ny = int(map_features.info.length_y/map_features.info.resolution)

        #normalize the terrain features to start at ego-height
        for k,v in zip(map_features.layers, map_features.data):
            #temp hack bc I don't like this feature.
#            if k not in ['roughness', 'height_high']:
#            if 'npts' in k:
            if False:
                continue
#                data = np.array(v.data).reshape(mf_nx, mf_ny)[::-1, ::-1]
#                data[data > 0] = np.log(data[data > 0])
#                map_feature_keys.append('log_' + k)
#                map_feature_data.append(data)

            else:
                data = np.array(v.data).reshape(mf_nx, mf_ny)[::-1, ::-1]

                map_feature_keys.append(k)
                map_feature_data.append(data)

        map_feature_data = np.stack(map_feature_data, axis=0)
        map_feature_data[~np.isfinite(map_feature_data) | (np.abs(map_feature_data) > 1e6)] = fill_value

#        start_time = map_features.info.header.stamp.to_sec()
        start_time = t.to_sec()
        targets = start_time + np.arange(horizon) * dt

        if targets.min() < timestamps.min():
            print('skipping...')
            continue

        if targets.max() > timestamps.max():
            print('done')
            break

        map_target_times.append(start_time)

        traj = traj_interp(targets)

        slim = 415.
        if len(steer_angles) > 0:
            steers = interp_steer(targets)
            if any(np.abs(steers) > slim):
                print("WARNING: steer {:.4f} exceeded the steer limit".format(max(abs(steers))))
            steers = np.clip(steers, -slim, slim) #clip to deal with potentially bad extrapolation

        if len(gps_poses) > 0:
            gps_subtraj = gps_interp(targets)

        if len(cmds) > 0:
            cmd_subtraj = np.stack([
                cmd_throttle_interp(targets),
                cmd_steer_interp(targets)
            ], axis=-1)

        #handle transforms to deserialize map/costmap
        map_metadata = map_features.info
        xmin = map_metadata.pose.position.x - 0.5 * (map_metadata.length_x)
        xmax = xmin + map_metadata.length_x
        ymin = map_metadata.pose.position.y - 0.5 * (map_metadata.length_y)
        ymax = ymin + map_metadata.length_y

        metadata_out = {
            'resolution': torch.tensor(map_metadata.resolution),
            'height': torch.tensor(map_metadata.length_x),
            'width': torch.tensor(map_metadata.length_y),
            'origin': torch.tensor([map_metadata.pose.position.x - 0.5 * (map_metadata.length_x), map_metadata.pose.position.y - 0.5 * (map_metadata.length_y)])
        }

        data = {
            'traj':torch.tensor(traj).float(),
            'cmd':torch.tensor(cmd_subtraj).float() if len(cmds) > 0 else None,
            'gps_traj':torch.tensor(gps_subtraj).float() if len(gps_poses) > 0 else None,
            'steer': torch.tensor(steers).float() if len(steer_angles) > 0 else None,
            'feature_keys': map_feature_keys,
            'map_features': torch.tensor(map_feature_data).float(),
            'metadata': metadata_out
        }

        data = {k:v for k,v in data.items() if v is not None}

        dataset.append(data)

        trajlen = data['traj'].shape[0]
        for k in ['traj', 'cmd', 'gps_traj', 'steer']:
            if k in data.keys():
                assert data[k].shape[0] == trajlen, 'SHAPE MISMATCH'
        i += 1

    #convert from gridmap to occgrid metadata
    if len(dataset) == 0:
        return None, None

    feature_keys = dataset[0]['feature_keys']
    dataset = {k:[x[k] for x in dataset] for k in dataset[0].keys() if k != 'feature_keys'}

#    dataset2 = {
#        'map_features':[x['map_features'] for x in dataset],
#        'traj':[x['traj'] for x in dataset],
#        'cmd':[x['cmd'] for x in dataset],
#        'gps_traj':[x['gps_traj'] for x in dataset],
#        'metadata':[x['metadata'] for x in dataset],
#        'steer':[x['steer'] for x in dataset]
#    }


    #If image topic exists, add to bag
    if image_topic is not None:
        bridge = CvBridge()
        image_timestamps = []
        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            image_timestamps.append(t.to_sec())
        image_timestamps = np.array(image_timestamps)
        #get closest image to targets
        dists = np.abs(np.expand_dims(image_timestamps, axis=0) - np.expand_dims(map_target_times, axis=1))
        image_targets = np.argmin(dists, axis=1)

        images = []
        for i, (topic, msg, t) in enumerate(bag.read_messages(topics=[image_topic])):
            n_hits = np.sum(image_targets == i)
            for j in range(n_hits):
#                img = np.frombuffer(msg.data, dtype=np.uint8)
                if msg._type == 'sensor_msgs/CompressedImage':
                    img = bridge.compressed_imgmsg_to_cv2(msg, "rgb8")
                else:
                    img = bridge.imgmsg_to_cv2(msg, "rgb8")
                img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
                images.append(torch.tensor(img).permute(2, 0, 1)[[2, 1, 0]] / 255.)

        dataset['image'] = images

    return dataset, feature_keys
