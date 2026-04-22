import jax
import jax.numpy as jnp

from functools import partial
from jax_tqdm import scan_tqdm


@partial(jax.jit, static_argnums=(0, 1, 2))
def hii(active_sites, number_batches, fn_gen, fn_args, z):
    '''Calculate the diagonal elements of the hessian of
    the input function evaluated at the pytree z.  The
    calculation splits the calculation into diagonal blocks
    of the hessian defined by the active_sites.  The diagonal
    of each of these blocks are calculated in a number of
    vmap'ed batches controlled by number_batches.

    For example if your function is a composite of several other
    functions that each take independent sets of variables the
    active_sites should be the collections of these independent
    sets.  E.g. if fn is a composite of three functions L, S, and
    A such that:

        fn(z) = L(z_1) + S(z_2, A(z_3))

    active_sites would be (z1, z2, z3) where each element is a
    tuple of variable names.  This kind of split vastly reduces
    the memory footprint of the calculation.

    Let's say in the previous example that the sets z1 and z2
    contain a small number of parameters (e.g. 10 each) but z3
    has a large number (e.g. 2500), you would want to control
    the number_batches for each set to be something like:

        number_batches = (1, 1, 40)

    This would calculate all of the z1 and z2 diags in parallel
    (with vmap), but batch z3 into about 40 sets of about 62 parameters
    with each set done in parallel (see docs for jax.lax.map).
    Note: this is the *number of batches* not the size of each batch.

    Parameters
    ----------
    active_sites : tuple of tuples
        Each inner tuple defines a block section of the hessian.
        This should be defined to take advantage of any natural
        symmetries of the function.
    number_batches : tuple
        The number of vmap'ed batches to use for each block defined
        by active_sites (must be same length as active_sites).
    fn : callable
        The function you want the diagonal of the hessian of.  Should
        be a callable that takes in the pytree z and returns a scalar.
    z : pytree
        Pytree containing the location the diagonal of the hessian
        should be calculated.
    '''
    def wrap_pot_fn(active):
        # split active sites from conditioned sites
        z_active = {k: v for k, v in z.items() if k in active}
        z_cond = {k: v for k, v in z.items() if k not in active}
        # flatten the active site and get unflattening function
        z_active_flat, unflat_active = jax.flatten_util.ravel_pytree(z_active)

        def cond_pot_fn_one_hot(i, x):
            # for active sites create a "one hot" function that fixes
            # all but the ith index of the input vector and evaluates
            # the pot_fn
            z_flat_update = z_active_flat.at[i].set(x)
            z = unflat_active(z_flat_update) | z_cond
            return fn_gen(fn_args)(z)
        return cond_pot_fn_one_hot, z_active_flat, unflat_active

    h = {}
    for active, batches in zip(active_sites, number_batches):
        pot_one_hot, z_active_flat, unflat_active = wrap_pot_fn(active)
        n = z_active_flat.shape[0]
        batch_size = n // batches
        idx = jnp.arange(n)

        def hess_fn(i):
            # function that finds the (i, i) component of the hessian matrix
            return jax.jacfwd(jax.jacfwd(pot_one_hot, argnums=1), argnums=1)(i, z_active_flat[i])
        # use map to loop over this function with a batch size to give control
        # over the memory footprint and the runtime tradeoff
        h_active_flat = jax.lax.map(hess_fn, idx, batch_size=batch_size)
        # unflatten the result and concat to the output
        h = h | unflat_active(h_active_flat)
    # return a pytree with same shape as z that has the diagonal of the hessian for each variable
    return h


@partial(jax.jit, static_argnums=(0, 1, 2))
def hii_scan(active_sites, number_batches, fn_gen, fn_args, zs):
    '''Calculate the diagonal elements of the hessian of
    the input function mapped over the leading axis of the pytree
    zs.  The calculation splits the calculation into diagonal
    blocks of the hessian defined by the active_sites.  The diagonal
    of each of these blocks are calculated in a number of
    vmap'ed batches controlled by number_batches.

    The mapping is done sequentially with jax.lax.scan to be
    memory efficient.  A progress bar is used to track the progress
    of this scan.

    For example if your function is a composite of several other
    functions that each take independent sets of variables the
    active_sites should be the collections of these independent
    sets.  E.g. if fn is a composite of three functions L, S, and
    A such that:

        fn(z) = L(z_1) + S(z_2, A(z_3))

    active_sites would be (z1, z2, z3) where each element is a
    tuple of variable names.  This kind of split vastly reduces
    the memory footprint of the calculation.

    Let's say in the previous example that the sets z1 and z2
    contain a small number of parameters (e.g. 10 each) but z3
    has a large number (e.g. 2500), you would want to control
    the number_batches for each set to be something like:

        number_batches = (1, 1, 40)

    This would calculate all of the z1 and z2 diags in parallel
    (with vmap), but batch z3 into about 40 sets of about 62 parameters
    with each set done in parallel (see docs for jax.lax.map).
    Note: this is the *number of batches* not the size of each batch.

    Parameters
    ----------
    active_sites : tuple of tuples
        Each inner tuple defines a block section of the hessian.
        This should be defined to take advantage of any natural
        symmetries of the function.
    number_batches : tuple
        The number of vmap'ed batches to use for each block defined
        by active_sites (must be same length as active_sites).
    fn : callable
        The function you want the diagonal of the hessian of.  Should
        be a callable that takes in the pytree z and returns a scalar.
    zs : pytree
        Pytree containing the locations the diagonal of the hessian
        should be calculated (mapped along the leading axis of each of
        the leaves).
    '''
    n = jax.tree.leaves(zs)[0].shape[0]
    idx = jnp.arange(n)

    @scan_tqdm(n, desc='Calculating hessian diagonal', tqdm_type='std')
    def one_z(_, input):
        _, z = input
        return None, hii(active_sites, number_batches, fn_gen, fn_args, z)
    _, hs = jax.lax.scan(one_z, None, (idx, zs))
    return hs
