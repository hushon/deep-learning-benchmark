import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'gpu'
# os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
# os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import jax
import jax.lax
import numpy as np
import jax.numpy as jnp

import torch
from torch.utils import data
from tqdm import trange, tqdm

import haiku as hk
import optax
import jmp

from haiku._src.nets.resnet import ResNet50

from typing import NamedTuple
import tree
import functools
import einops

if jax.default_backend() == 'gpu':
    N_DEVICES = jax.local_device_count()
else:
    N_DEVICES = 1

class FLAGS(NamedTuple):
    KEY = jax.random.PRNGKey(1)
    BATCH_SIZE = 1024
    MAX_EPOCH = 100
    INIT_LR = 2e-1
    N_WORKERS = 2
    USE_AMP = True
    NET_CLASS = ResNet50

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DummyImagenet(data.Dataset):
    def __init__(self):
        super().__init__()
        self.size = 1024
        self.input = torch.randint(0, 255, (self.size, 3, 224, 224), dtype=torch.uint8, generator=torch.Generator().manual_seed(42))
        self.target = torch.randint(0, 1000, size=(self.size, ), generator=torch.Generator().manual_seed(42))

    def __getitem__(self, idx):
        i = idx % self.size
        x = self.input[i]
        y = self.target[i]
        return (x, y)

    def __len__(self):
        return 1281167


def tprint(obj):
    tqdm.write(obj.__str__())


class TrainState(NamedTuple):
    params: hk.Params
    state: hk.State
    opt_state: optax.OptState
    loss_scale: jmp.LossScale


def softmax_cross_entropy(logits, labels):
    logp = jax.nn.log_softmax(logits)
    loss = -jnp.take_along_axis(logp, labels[:, None], axis=-1)
    return loss



def l2_loss(params):
    l2_params = [p for (mod_name, _), p in tree.flatten_with_path(
        params) if 'batchnorm' not in mod_name]
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in l2_params)


def forward(images, is_training: bool):
    net = FLAGS.NET_CLASS(num_classes=1000)
    return net(images, is_training=is_training)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def icycle(iterable):
    while True:
        for x in iterable:
            yield x


def main():
    dataloader = MultiEpochsDataLoader(
        DummyImagenet(),
        batch_size=FLAGS.BATCH_SIZE,
        num_workers=0,
        pin_memory=False,
        shuffle=True,
        drop_last=True,
    )
    dataloader_iter = icycle(dataloader)

    ## MIXED PRICISION ##
    if FLAGS.USE_AMP:
        mp_policy = jmp.Policy(param_dtype = jnp.float32,
                                compute_dtype = jnp.float16,
                                output_dtype = jnp.float32)
        mp_bn_policy = jmp.Policy(param_dtype = jnp.float32,
                                compute_dtype = jnp.float32,
                                output_dtype = jnp.float16)
        loss_scale = jmp.DynamicLossScale(2.0**15)
    else:
        mp_policy = jmp.Policy(param_dtype = jnp.float32,
                                compute_dtype = jnp.float32,
                                output_dtype = jnp.float32)
        mp_bn_policy = jmp.Policy(param_dtype = jnp.float32,
                                compute_dtype = jnp.float32,
                                output_dtype = jnp.float32)
        loss_scale = jmp.NoOpLossScale()

    hk.mixed_precision.set_policy(FLAGS.NET_CLASS, mp_policy)
    hk.mixed_precision.set_policy(hk.BatchNorm, mp_bn_policy)


    ## MODEL TRANSFORM ##
    model = hk.transform_with_state(forward)
    # model = hk.without_apply_rng(model)

    ## OPTIMIZER ##
    learning_rate_fn = optax.cosine_onecycle_schedule(
        transition_steps=len(dataloader)*FLAGS.MAX_EPOCH,
        peak_value=FLAGS.INIT_LR,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4
    )
    optimizer = optax.sgd(learning_rate_fn, momentum=0.9, nesterov=False)


    ## TRAIN STATE ##
    def get_init_trainstate(rng_key):
        params, state = model.init(rng_key, jnp.empty((1, 224, 224, 3), dtype=jnp.float32), is_training=True)
        opt_state = optimizer.init(params)
        train_state = TrainState(params, state, opt_state, loss_scale)
        return train_state


    train_state = get_init_trainstate(FLAGS.KEY)
    train_state = jax.device_put_replicated(train_state, jax.local_devices())


    @functools.partial(jax.pmap, axis_name='i', donate_argnums=(0,))
    def train_step(train_state: TrainState, batch: dict):
        params, state, opt_state, loss_scale = train_state
        input, target = batch['image'], batch['label']
        input = input.astype(jnp.float32) / 255.0
        mean = jnp.array(IMAGENET_MEAN).reshape(1, 1, 1, -1)
        std = jnp.array(IMAGENET_STD).reshape(1, 1, 1, -1)
        input = (input - mean) / std
        def loss_fn(p):
            logits, state_new = model.apply(
                p, state, FLAGS.KEY, input, is_training=True)
            ce_loss = softmax_cross_entropy(logits, target).mean()
            loss = ce_loss + 1e-4 * l2_loss(p)
            scaled_loss = loss_scale.scale(loss)
            return scaled_loss, (loss, logits, state_new)
        grads, aux = jax.grad(loss_fn, has_aux=True)(params)
        loss, logits, state = aux
        grads = mp_policy.cast_to_compute(grads)
        grads = loss_scale.unscale(grads)
        grads = jax.lax.pmean(grads, axis_name='i')
        grads = mp_policy.cast_to_param(grads)
        deltas, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, deltas)
        if FLAGS.USE_AMP:
            grads_finite = jmp.all_finite(grads)
            loss_scale = loss_scale.adjust(grads_finite)
            train_state_new = TrainState(params, state, opt_state, loss_scale)
            train_state_new = jmp.select_tree(
                                            grads_finite,
                                            train_state_new,
                                            train_state
                                            )
        else:
            train_state_new = TrainState(params, state, opt_state, loss_scale)
        return train_state_new, loss, logits


    max_idx = len(dataloader) * FLAGS.MAX_EPOCH
    pbar = trange(max_idx)

    for iter_idx in pbar:
        input, target = next(dataloader_iter)
        input = einops.rearrange(np.asarray(input), '(n b) h w c -> n b h w c', n=N_DEVICES)
        target = einops.rearrange(np.asarray(target), '(n b) -> n b', n=N_DEVICES)
        batch = {
            'image': input,
            'label': target,
        }
        train_state, loss, logits = train_step(train_state, batch)

        if iter_idx>0:
            pbar.set_description(f"{FLAGS.BATCH_SIZE*pbar.format_dict['rate']:.2f} samples/sec")

        if iter_idx%100 == 0:
            loss = np.mean(loss)
            tqdm.write(f'Train loss: {loss.item():.4f}')


if __name__ == '__main__':
    main()
