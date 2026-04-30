import numpy as np 
import jax.numpy as jnp
import jax
import distrax
import optax
import chex

def low_rank_model(D,K):
    alpha=2.0
    beta=2.0
    U = distrax.Laplace(0.0,1.0)._sample_n(key=jax.random.PRNGKey(72), n=D)
    U = U.reshape((D,1))
    A = U@U.T
    
    b = distrax.Gamma(2.0,2.0).sample(seed=jax.random.PRNGKey(72)) * jnp.ones([D, 1])
    b = b.reshape((D,1))
    chex.assert_shape(A,(D, D))
    chex.assert_shape(U,(D,1))
    chex.assert_shape(b,(D, 1))
    @jax.vmap
    def log_pdf(x, y):
        chex.assert_shape(x.T@U@U.T@b,(1,))
        chex.assert_shape(y,(1,))
        return jnp.squeeze(distrax.Bernoulli(jax.nn.sigmoid(x.T@U@U.T@b)).log_prob(y))
    def sample(prng, bsz):
        prng, subprng = jax.random.split(prng)
        x = distrax.Normal(0.0,1.0)._sample_n(subprng, bsz*D)
        x = x.reshape((bsz, D, 1))
        subprng = jax.random.split(prng, bsz)
        y = jax.vmap(lambda _x, pr: distrax.Bernoulli(jax.nn.sigmoid(_x.T@U@U.T@b)).sample(seed=pr), in_axes=(0,0))(x, subprng)
        return jnp.squeeze(x,axis=-1),jnp.squeeze(y, axis=-1)
    return log_pdf, sample

logpdf, sample = low_rank_model(100,1)
prng = jax.random.PRNGKey(42)
x,y = sample(prng, 1000)
import matplotlib.pyplot as plt  

import pandas as pd
x = jnp.concatenate([x,y],axis=-1)
my_df = pd.DataFrame(np.asarray(x))
my_df.columns = [f'x{i}' for i in range(100)]+['y']
my_df.to_csv('regression.csv', index=False)