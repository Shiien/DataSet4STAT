import numpy as np 
import jax.numpy as jnp
import jax
import distrax
import optax

def low_rank_model(D,K):

    alpha=2.0
    beta=2.0
    U = distrax.Laplace(0.0,1.0)._sample_n(key=jax.random.PRNGKey(72), n=D)
    A = U@U.T
    b = distrax.Gamma(2.0,2.0).sample(seed=jax.random.PRNGKey(72)) * jnp.ones([D])

    @jax.vmap
    def log_pdf(x, y):
        return distrax.Bernoulli(jax.nn.sigmoid(jnp.sum(x*U * U*b,axis=-1))).log_prob(y)
    def sample(prng,bsz):
        prng, subprng = jax.random.split(prng)
        x = distrax.Normal(0.0,1.0)._sample_n(subprng, bsz*D)
        x = x.reshape((bsz, -1))
        prng, subprng = jax.random.split(prng)
        y = distrax.Bernoulli(jax.nn.sigmoid(jnp.sum(x*U * U*b,axis=-1))).sample(seed=subprng)
        return x,y
    return log_pdf, sample

logpdf, sample = low_rank_model(100,1)
prng = jax.random.PRNGKey(42)
x,y = sample(prng, 1000)
import matplotlib.pyplot as plt  
# print(x.shape,y.shape)
# print(x,y,logpdf(x,y))

import numpy
import pandas as pd
print(x.shape,y.shape)
x = jnp.concatenate([x,y.reshape((1000,1))],axis=-1)
my_df = pd.DataFrame(np.asarray(x))
my_df.columns = [f'x{i}' for i in range(100)]+['y']
my_df.to_csv('regression.csv', index=False)