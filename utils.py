from kerassurgeon import Surgeon
import numpy as np
from matplotlib import pyplot as plt

def prune_model(model, pruning_percent_step, pruning_standart_deviation_part):
    idxs = []
    count = len(model.layers)
    for layer_idx in range(count):
        name = model.layers[layer_idx].name
        if(name.startswith("conv")):
            w1 = model.layers[layer_idx].get_weights()[0]
            weight = w1
            weights_dict = {}
            num_filters = len(weight[0,0,0,:])
            idxs.append(layer_idx)
    
    idxs.reverse()

    surgeon = Surgeon(model)
    for layer_idx in idxs:
        layer = model.layers[layer_idx]
        w1 = model.layers[layer_idx].get_weights()[0]
        weight = w1
        weights_dict = {}
        num_filters = len(weight[0,0,0,:])
        # print(num_filters)
        delete_part_int = (int)(num_filters * pruning_percent_step) + 1

        l1_s = []
        for j in range(num_filters):
            l1 = np.sum(abs(weight[:,:,:,j]))
            filt = f"{j}"
            weights_dict[filt] = l1
            l1_s.append(l1)

        weights_dict_sort = sorted(weights_dict.items(), key = lambda kv: kv[1])
        mean = np.mean(l1_s)
        std = np.std(l1_s)

        threshold_l1 = mean - std*pruning_standart_deviation_part
        delete_idxs = []
        for i in range(delete_part_int):
            if(weights_dict_sort[i][1] > threshold_l1):
                break

            delete_idxs.append((int)(weights_dict_sort[i][0]))
        
        surgeon.add_job(job = "delete_channels", layer=layer, channels=delete_idxs,)
    pruned_model = surgeon.operate()

    return pruned_model

def plot_and_save_stats(stats):
    sizes = []
    losses = []
    times = []
    x = []
    for i in range(len(stats)):
        stat = stats[i]

        x.append(i)
        sizes.append(stat["size"])
        losses.append(stat["loss"])
        times.append(stat["time"])

    sizes /= np.max(sizes)
    losses /= np.max(losses)
    times /= np.max(times)

    fig = plt.figure()
    size_line, = plt.plot(x, sizes, color='red', lw=1, label='Size', linestyle='--')
    loss_line, = plt.plot(x, losses, color='green', lw=1, label='Loss', linestyle=':')
    inf_line, = plt.plot(x, times, color='blue', lw=1, label='Inference time')
    plt.legend(handles=[size_line, loss_line, inf_line])

    plt.xlabel('Steps')
    plt.grid(True)

    plt.savefig("pruning_res.png")

