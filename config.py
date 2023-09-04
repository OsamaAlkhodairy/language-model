warmup_iters = 2000
max_iters = 60000

save_model_iters = 2000

max_learning_rate = 1e-5
min_learning_rate = 1e-6

batch_size = 64
block_size = 128
n_embed = 384
eval_interval = 1000
save_model_iters = 1000
eval_iters = 200
dropout = 0.1
n_head = 8
n_layers = 8

# print('max iters from config', max_iters)

print('The following are the hyperparameters')
def print_globals():
    global_vars = globals()
    for var_name, var_value in global_vars.items():
        print(f'{var_name} = {var_value}')

print_globals()