from dataset import CIFAR10_wrapper
from networks import ResnetWrapper
import pytorch_lightning as pl
from parametrization import get_n_parameters

effective_batch_size = 10
epochs = 4
precision = 32
# in the parametrized networks
# by relaxing this condition the actual N of parameters will get closer to the number of those of the original network
max_units_per_mlp_layer = 250
n_hidden_layers = 5
save_folder = 'logs_and_models/'
# save_folder = '/media/ramdisk/'

data = CIFAR10_wrapper(
    data_location='/media/ramdisk',
    batch_size=effective_batch_size,
    workers=15,
)
data_parametrized = CIFAR10_wrapper(
    data_location='/media/ramdisk',
    batch_size=1,
    workers=15,
)
non_parametrized = ResnetWrapper(parametrize=False)
# parametr_nonp = get_n_parameters(non_parametrized)
# print(f'Non-parametrized has {parametr_nonp} parameters')
parametrized = ResnetWrapper(parametrize=True,
                             max_units_per_mlp_layer=max_units_per_mlp_layer,
                             n_hidden_layers=n_hidden_layers
                             )
# parametr_p = get_n_parameters(parametrized)
# print(f'Parametrized has {parametr_p} parameters')
# difference=parametr_nonp - parametr_p
# print(f'Difference: {difference}, {int(difference/parametr_nonp * 100)}% of the non-parametrized networks')
print('The number of parameters was limited to keep the network manageable for my GPU')
print(f'Networks will be trained with a batch size of {effective_batch_size} for {epochs} epochs')


trainer_nopar = pl.Trainer(
    default_root_dir=f'{save_folder}trainer_logs_nonpar',
    precision=precision,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath=f'{save_folder}checkpoints_nopar',
            filename='best',
            monitor='Validation loss',
            verbose=True,
            save_last=True,
            every_n_epochs=1,
            mode='min',
        )
    ],
    max_time={'hours': 1},
    accelerator='gpu',
    devices=1,
    max_epochs=epochs,
)

trainer_par = pl.Trainer(
    default_root_dir=f'{save_folder}trainer_logs_par',
    precision=precision,
    accumulate_grad_batches=effective_batch_size,
    callbacks=[
        pl.callbacks.ModelCheckpoint(
            dirpath=f'{save_folder}checkpoints_par',
            filename='best',
            monitor='Validation loss',
            verbose=True,
            save_last=True,
            every_n_epochs=1,
            mode='min',
        )
    ],
    accelerator='gpu',
    devices=1,
    max_epochs=epochs,
)

print('Fitting parametrized network')
trainer_par.fit(parametrized, data_parametrized)

print('Fitting non-parametrized network')
trainer_nopar.fit(non_parametrized, data)

print("Testing parametrized network")
trainer_par.test(parametrized, data_parametrized)

print("Testing non-parametrized network")
trainer_nopar.test(non_parametrized, data)
