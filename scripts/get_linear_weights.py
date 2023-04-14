"""
Look at the linear weights of a model
"""

import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fp', type=str, required=True, help='path to the model')
    args = parser.parse_args()

    model = torch.load(args.model_fp)

    weights = [w.weight.squeeze().detach() for w in model.network.cost_heads]
    weights = torch.stack(weights, dim=0)

    fks = model.expert_dataset.feature_keys

    z_scores = (weights.mean(dim=0)/weights.std(dim=0)).abs()

    for i in torch.argsort(z_scores, descending=True):
        fk = fks[i]
        ws = weights[:, i]
        print('{:<20}:\t{:.2f} += {:.2f} (z={:.2f})'.format(fk, ws.mean(), ws.std(), ws.mean()/ws.std()))
