import torch

def quat_to_yaw(quat):
    """
    Convert a quaternion (as [x, y, z, w]) to yaw
    """
    if len(quat.shape) < 2:
        return quat_to_yaw(quat.unsqueeze(0)).squeeze()

    return torch.atan2(2 * (quat[:, 3]*quat[:, 2] + quat[:, 0]*quat[:, 1]), 1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2))    

def get_speedmap(trajs, map_metadata, weights=None):
    """
    Given a set of trajectories, produce a map where each cell contains the speed the traj in that cell
    Args:
        trajs: the trajs to compute speeds over
        map_metadata: The map params to get speeds over
        weights: optional weighting on each trajectory
    """
    assert trajs.shape[-1] == 13, 'This method only works with the 13d [p, q, v, w] state formulation'
    if weights is None:
        weights = torch.ones(trajs.shape[0], device=trajs.device) / trajs.shape[0]

    xs = trajs[...,0]
    ys = trajs[...,1]
    res = map_metadata['resolution']
    ox = map_metadata['origin'][0].item()
    oy = map_metadata['origin'][1].item()
    nx = round(map_metadata['width'] / res)
    ny = round(map_metadata['height'] / res)
    width = max(nx, ny)

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    # 1 iff. valid
    valid_mask = (xidxs >= 0) & (xidxs < ny) & (yidxs >= 0) & (yidxs < nx)

    #I'm pretty sure all I have to do is replace the ones from svs w/ the actual speeds
    speeds = torch.linalg.norm(trajs[...,7:10], axis=-1)
    binweights = torch.ones(trajs.shape[:-1], device=trajs.device) * weights.view(-1, 1) * valid_mask.float()
    speed_binweights = speeds * binweights

    flat_binweights = binweights.flatten()
    flat_speed_binweights = speed_binweights.flatten()

    flat_speeds = (nx * yidxs + xidxs).flatten().clamp(0, nx*ny - 1).long()
    flat_speed_counts = torch.bincount(flat_speeds, weights=flat_speed_binweights)

    flat_visits = (nx * yidxs + xidxs).flatten().clamp(0, nx*ny - 1).long()
    flat_visit_counts = torch.bincount(flat_visits, weights=flat_binweights) + 1e-6

    bins = torch.zeros(nx*ny, device=trajs.device)
    bins[:len(flat_speed_counts)] += flat_speed_counts / flat_visit_counts
    speed_counts = bins.view(nx, ny)

    #debug
#    with torch.no_grad():
#        idx_diffs = (((nx * yidxs + xidxs)[..., 1:] - (nx * yidxs + xidxs)[..., :-1]).abs() > 1e-4).float()
#
#        import matplotlib.pyplot as plt
#        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
#        axs = axs.flatten()
#        for traj, sp, diff in zip(trajs, speeds, idx_diffs):
#            axs[0].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='b', alpha=0.5)
#            axs[2].plot(sp.cpu())
#            axs[2].scatter(torch.arange(len(diff)) + 1, diff.cpu() * sp[1:].cpu(), c='r', s=2.)
#        axs[0].imshow(speed_counts.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
#        axs[1].imshow(speed_counts.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
#        plt.show()

    return speed_counts

def get_state_visitations(trajs, map_metadata, weights = None):
    """
    Given a set of trajectories and map metadata, compute the visitations on the map for each traj.
    Args:
        trajs: The trajs to get visit counts of
        map_metadata: The map parameters to get visit counts from
        weights: (optional) the amount to weight each traj by
    """
    if weights is None:
        weights = torch.ones(trajs.shape[0], device=trajs.device) / trajs.shape[0]

    xs = trajs[...,0]
    ys = trajs[...,1]
    res = map_metadata['resolution']
    ox = map_metadata['origin'][0].item()
    oy = map_metadata['origin'][1].item()
    nx = round((map_metadata['width'] / res))
    ny = round((map_metadata['height'] / res))
    width = max(nx, ny)

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    # 1 iff. valid
    valid_mask = (xidxs >= 0) & (xidxs < ny) & (yidxs >= 0) & (yidxs < nx)

    binweights = torch.ones(trajs.shape[:-1], device=trajs.device) * weights.view(-1, 1) * valid_mask.float()
    flat_binweights = binweights.flatten()

    flat_visits = (nx * yidxs + xidxs).flatten().clamp(0, nx*ny - 1).long()
    flat_visit_counts = torch.bincount(flat_visits, weights=flat_binweights)
    bins = torch.zeros(nx*ny, device=trajs.device)
    bins[:len(flat_visit_counts)] += flat_visit_counts
    visit_counts = bins.view(nx, ny)
    visitation_probs = visit_counts / visit_counts.sum()

    #debug
#    with torch.no_grad():
#        import matplotlib.pyplot as plt
#        fig, axs = plt.subplots(1, 2)
#        for traj in trajs:
#            axs[0].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='b', alpha=0.1)
#        axs[0].imshow(visitation_probs.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
#        axs[1].imshow(visitation_probs.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
#        plt.show()

    return visitation_probs

def batch_get_state_visitations(trajs, map_metadata, weights = None):
    """
    Given a set of trajectories and map metadata, compute the visitations on the map for each traj.
    Args:
        trajs: The trajs to get visit counts of
        map_metadata: The map parameters to get visit counts from
        weights: (optional) the amount to weight each traj by

    Unlike regular get_state_visitations, expect a leading batch dim on trajs [B x ... X T x m]
    and a leading batch dim on the origin of metadata [B x 2]
    """
    if weights is None:
        weights = torch.ones(trajs.shape[0], device=trajs.device) / trajs.shape[0]

    xs = trajs[...,0]
    ys = trajs[...,1]
    res = map_metadata['resolution']
    ox = map_metadata['origin'][..., 0]
    oy = map_metadata['origin'][..., 1]
    nx = round(map_metadata['width'] / res)
    ny = round(map_metadata['height'] / res)
    width = max(nx, ny)

    n_trailing_dims = len(xs.shape)-1
    origin_shape = [ox.shape[0]] + [1] * n_trailing_dims

    xidxs = ((xs - ox.view(*origin_shape)) / res).long()
    yidxs = ((ys - oy.view(*origin_shape)) / res).long()

    # 1 iff. valid
    valid_mask = (xidxs >= 0) & (xidxs < ny) & (yidxs >= 0) & (yidxs < nx)

    binweights = torch.ones(trajs.shape[:-1], device=trajs.device) * weights.view(-1, 1) * valid_mask.float()
    flat_binweights = binweights.flatten()

    flat_visits = (nx * yidxs + xidxs).flatten().clamp(0, nx*ny - 1).long()
    flat_visit_counts = torch.bincount(flat_visits, weights=flat_binweights)
    bins = torch.zeros(nx*ny, device=trajs.device)
    bins[:len(flat_visit_counts)] += flat_visit_counts
    visit_counts = bins.view(nx, ny)
    visitation_probs = visit_counts / visit_counts.sum()

    #debug
#    with torch.no_grad():
#        import matplotlib.pyplot as plt
#        fig, axs = plt.subplots(1, 2)
#        for traj in trajs:
#            axs[0].plot(traj[:, 0].cpu(), traj[:, 1].cpu(), c='b', alpha=0.1)
#        axs[0].imshow(visitation_probs.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
#        axs[1].imshow(visitation_probs.cpu(), origin='lower', extent=(ox, ox+map_metadata['width'], oy, oy+map_metadata['height']))
#        plt.show()

    return visitation_probs

def get_feature_counts(traj, map_features, map_metadata):
    """
    Given a (set) of trajectories and map features, compute the features of that trajectory
    """
    xs = traj[...,0]
    ys = traj[...,1]
    res = map_metadata['resolution']
    ox = map_metadata['origin'][0]
    oy = map_metadata['origin'][1]

    xidxs = ((xs - ox) / res).long()
    yidxs = ((ys - oy) / res).long()

    valid_mask = (xidxs >= 0) & (xidxs < map_features.shape[2]) & (yidxs >= 0) & (yidxs < map_features.shape[1])

    xidxs[~valid_mask] = 0
    yidxs[~valid_mask] = 0

    # map data is transposed
    features = map_features[:, yidxs, xidxs]

    feature_counts = features.mean(dim=-1)
    return feature_counts

def dict_to(d1, device):
    if isinstance(d1, dict):
        return {k:dict_to(v, device) for k,v in d1.items()}
    else:
        return d1.to(device)
