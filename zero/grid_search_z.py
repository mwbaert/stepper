import os

# HYPERPARAMETERS
gamma = [0.9]
batch_size = [256]
mom = [0.8]
exp_replay = [20000]
eps_decay = [100000, 500000]
lr = [0.001, 0.01]
neurons = [64, 128, 256]
episodes = 1800 #TODO aanpassen naar een hoger aantal

# GRID SEARCH
exp_count = 0
total_exp = len(gamma) * len(batch_size) * len(mom) * len(exp_replay) * len(eps_decay) * len(lr) * len(neurons)
script = "sm_z_7.py"


for G in gamma:
    for BS in batch_size:
        for M in mom:
            for EX in exp_replay:
                for EP in eps_decay:
                    for L in lr:
                        for N in neurons:
                            print("experiment {}/{}".format(exp_count+1, total_exp))
                            # TODO to run on ELIS GPU type: CUDA_VISIBLE_DEVICE=[Xgpu]
                            os.system("python "
                                        + script
                                        + " --gamma={} --batch_size={} --mom={} --exp_replay={} --eps_decay={} "
                                          "--lr={} --neurons={} --folder={} --episodes={}"
                                        .format(G, BS, M, EX, EP, L, N, exp_count, episodes))
                            exp_count += 1
