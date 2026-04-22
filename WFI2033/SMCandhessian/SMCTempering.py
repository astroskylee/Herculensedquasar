import jax
import jax.numpy as jnp
import jax_tqdm

import numpyro
import numpyro.infer.util as util

import optimistix as optx

from .hessian_daig_gen import hii_scan
from functools import partial
from collections import namedtuple
from numpyro.infer.hmc import HMCState
from functools import partial

# helper functions


def split_prior_likelihood(model, model_args, model_kwargs, params):
    # given an unconstrained tree of params return the log probability of the prior and likelihood as two values
    substituted_model = numpyro.handlers.substitute(
        model, substitute_fn=partial(util._unconstrain_reparam, params)
    )
    log_probs, trace = numpyro.infer.util.compute_log_probs(substituted_model, model_args, model_kwargs, {})
    prior_keys = [
        key for key, value in trace.items()
        if value['type'] == 'sample' and not value['is_observed']
    ]
    # associate any constrained transforms' log_det along side the value
    prior_values = [log_probs[k] + log_probs.get(f'_{k}_log_det', 0.0) for k in prior_keys]
    likelihood_keys = [
        key for key, value in trace.items()
        if value['type'] == 'sample' and value['is_observed'] and not value['infer'].get('is_auxiliary', False)
    ]
    # associate any constrained transforms' log_det along side the value
    likelihood_values = [log_probs[k] + log_probs.get(f'_{k}_log_det', 0.0) for k in likelihood_keys]
    return sum(prior_values, start=0.0), sum(likelihood_values, start=0.0)


def potential_fn_temp(model, model_args, model_kwargs, beta, params):
    log_prior, log_likelihood = split_prior_likelihood(model, model_args, model_kwargs, params)
    return -(log_prior + beta * log_likelihood)


def log_mean(log_value):
    return jax.nn.logsumexp(log_value) - jnp.log(log_value.size)


def log_norm(log_value):
    return log_value - jax.nn.logsumexp(log_value)


def log_weights(log_likelihood, beta1, beta2):
    return (beta2 - beta1) * log_likelihood


def ess(log_weights):
    log_numerator = 2 * jax.nn.logsumexp(log_weights)
    log_denominator = jax.nn.logsumexp(2 * log_weights)
    return jnp.exp(log_numerator - log_denominator)


def root_fn(beta2, args):
    log_likelihood, current_beta, alpha = args
    w = log_weights(log_likelihood, current_beta, beta2)
    return ess(w) - (alpha * w.shape[0])


solver = optx.Bisection(rtol=1e-5, atol=1e-5)


def get_next_beta(log_likelihood, current_beta, alpha):
    # def root_fn(beta2, alpha):
    #     w = log_weights(log_likelihood, current_beta, beta2)
    #     return ess(w) - (alpha * w.shape[0])
    sol = optx.root_find(
        root_fn, solver,
        current_beta,
        args=(log_likelihood, current_beta, alpha),
        options={'lower': current_beta, 'upper': 5.0}
    )
    return sol.value


def systematic_samp(rng_key, num_samples, weights):
    n = weights.shape[0]
    u = jax.random.uniform(rng_key, ())
    cumsum = jnp.cumsum(weights)
    linspace = (jnp.arange(num_samples, dtype=weights.dtype) + u) / num_samples
    idx = jnp.searchsorted(cumsum, linspace)
    return jnp.clip(idx, 0, n-1)


def pytree_select(idx, z):
    return jax.tree.map(lambda x: x[idx], z)


SMCTemperingState = namedtuple(
    'SMCTemperingState',
    [
        'count',
        'x',
        'z',
        'z_filter',
        'beta',
        'beta_next',
        'rng_key',
        'log_prior',
        'log_likelihood',
        'log_z_ratio',
        'log_z_estimate',
        'SamplerState',
        'StepSizeState',
        'inverse_mass_matrix'
    ]
)


class SMCTempering():
    def __init__(
        self,
        model,
        model_args=(),
        model_kwargs={},
        alpha=0.5,
        draw_size=2000,
        num_particles=50,
        num_warmup=100,
        first_step_size=1,
        nuts_kwargs={},
        inv_mass_max=1e4,
        hessian_diag_args=(),
        divergent_restart=True,
        resample_chains=True
    ):
        self.model = model
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.alpha = alpha
        self.draw_size = draw_size
        self.num_particles = num_particles
        self.num_warmup = num_warmup
        self.num_samples = self.draw_size // self.num_particles
        self.nuts_kwargs = nuts_kwargs
        self.first_step_size = first_step_size

        # make a new instance of NUTS that uses a custom _potential_fn_gen
        self.inv_mass_max = inv_mass_max
        self.ss_i, self.ss_u = numpyro.infer.hmc_util.dual_averaging()
        self.hessian_diag_args = hessian_diag_args
        self.NUTS_init = False
        self.divergent_restart = divergent_restart
        self.resample_chains = resample_chains
        # self.hii_scan = jax.jit(partial(
        #     hii_scan,
        #     *hessian_diag_args
        # ))

    def make_sampler(self, adapt_step=True, step_size=1, warmup=True):
        # make a new instance of NUTS that uses a custom _potential_fn_gen
        self.inner_kernel = numpyro.infer.NUTS(
            potential_fn=self._potential_fn_gen,
            step_size=step_size,
            adapt_step_size=adapt_step,
            adapt_mass_matrix=False,
            **self.nuts_kwargs
        )
        # perhaps this is the only thing that needs to be replaced
        # to "reset"
        nuts_init_fn, nuts_sample_fn = numpyro.infer.hmc.hmc(
            potential_fn_gen=self._potential_fn_gen,
            kinetic_fn=numpyro.infer.hmc_util.euclidean_kinetic_energy,
            algo="NUTS"
        )
        self.inner_kernel._potential_fn = None
        self.inner_kernel._potential_fn_gen = self._potential_fn_gen
        self.inner_kernel._init_fn = nuts_init_fn
        self.inner_kernel._sample_fn = nuts_sample_fn
        if warmup:
            num_warmup = self.num_warmup
        else:
            num_warmup = 0
        self.mcmc = numpyro.infer.MCMC(
            self.inner_kernel,
            num_warmup=num_warmup,
            num_chains=self.num_particles,
            num_samples=self.num_samples,
            postprocess_fn=self._post_process,
            chain_method='vectorized',
            jit_model_args=True  # needs to be True to be re-used without issue
        )

    def constrain_one(self, z):
        return numpyro.infer.util.constrain_fn(
            self.model,
            self.model_args,
            self.model_kwargs,
            z,
            return_deterministic=True
        )

    def constrain(self, zs):
        return jax.lax.map(self.constrain_one, zs)

    def _potential_fn_gen(self, beta):
        # to add Gibbs add a z_fixed arg that is passed in
        def pe_fn(z):
            return potential_fn_temp(
                self.model,
                self.model_args,
                self.model_kwargs,
                beta,
                z
            )
        return pe_fn

    def _post_process(self, params):
        log_prior, log_likelihood = split_prior_likelihood(
            self.model,
            self.model_args,
            self.model_kwargs,
            params
        )
        output = params.copy()  # important to return a copy rather than mutate the input
        output['_log_likelihood'] = log_likelihood
        output['_log_prior'] = log_prior
        return output

    def draw_from_prior(self, rng_key):
        prototype_params = util.find_valid_initial_params(
            rng_key,
            self.model,
            model_args=self.model_args,
            model_kwargs=self.model_kwargs,
            init_strategy=numpyro.infer.init_to_sample(),
            validate_grad=False
        )[0][0]
        draw_key, rng_key = jax.random.split(rng_key)
        draw_keys = jax.random.split(draw_key, self.draw_size)

        @jax_tqdm.scan_tqdm(self.draw_size, desc='Draw from prior', tqdm_type='std')
        def get_one_z(_, input):
            _, rng_key = input
            z = util.find_valid_initial_params(
                rng_key,
                self.model,
                model_args=self.model_args,
                init_strategy=numpyro.infer.init_to_sample(),
                model_kwargs=self.model_kwargs,
                validate_grad=False,
                prototype_params=prototype_params
            )[0][0]
            log_prior, log_like = split_prior_likelihood(
                self.model,
                self.model_args,
                self.model_kwargs,
                z
            )
            return None, (log_prior, log_like, z)

        _, (log_prior, log_likelihood, z) = jax.lax.scan(
            get_one_z, None, (jnp.arange(self.draw_size), draw_keys)
        )
        state = SMCTemperingState(
            count=0,
            x=self.constrain(z),
            z=z,
            z_filter=None,
            beta=0.0,
            beta_next=None,
            rng_key=rng_key,
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            log_z_ratio=0.0,
            log_z_estimate=0.0,
            SamplerState=None,
            StepSizeState=None,
            inverse_mass_matrix=None
        )
        return self.update_smc(state)

    def soft_abs_inv(self, x):
        return jnp.tanh(self.inv_mass_max * x) / x

    def update_smc(self, state):
        resample_key, rng_key = jax.random.split(state.rng_key, 2)
        if state.beta == 1.0:
            beta_next = 1.0
        else:
            beta_next = get_next_beta(state.log_likelihood, state.beta, self.alpha)
            beta_next = jnp.clip(beta_next, max=1.0)
        log_w = log_weights(state.log_likelihood, state.beta, beta_next)
        log_z_ratio = log_mean(log_w)
        if self.resample_chains:
            log_W = log_norm(log_w)
            W = jnp.exp(log_W)
            idx = systematic_samp(resample_key, self.num_particles, W)
        else:
            # select the last point for each chain so it picks up where it left off
            idx = jnp.arange(self.num_particles) * self.num_samples + (self.num_samples - 1)

        resamp_z = pytree_select(idx, state.z)
        diag_hessian = hii_scan(
            *self.hessian_diag_args,
            self._potential_fn_gen,
            beta_next,
            resamp_z
        )

        inverse_mass_matrix = jax.tree.map(
            self.soft_abs_inv, diag_hessian
        )

        inv_mass_matrix_flat = jax.vmap(
            lambda x: jax.flatten_util.ravel_pytree(x)[0]
        )(inverse_mass_matrix)
        print('----')
        return state._replace(
            count=state.count + 1,
            z_filter=resamp_z,
            beta_next=beta_next,
            rng_key=rng_key,
            log_z_ratio=log_z_ratio,
            log_z_estimate=state.log_z_estimate + log_z_ratio,
            inverse_mass_matrix=inv_mass_matrix_flat
        )

    def init_nuts(self, state):
        # The first time NUTS is called the `mcmc.sampler.init`
        # must be called so the vectorization works as intended
        #
        # ONLY CALL THIS ONCE
        # use the divergent_update or replace_nuts if updates are
        # needed after the first call
        init_key, rng_key = jax.random.split(state.rng_key, 2)
        inv_mass_flat_sqrt = jnp.sqrt(state.inverse_mass_matrix)
        mass_flat_sqrt = 1 / inv_mass_flat_sqrt
        NUTS_state = self.mcmc.sampler.init(
            jax.random.split(init_key, self.num_particles),
            self.mcmc.num_warmup,
            init_params=state.z_filter,
            model_args=(state.beta_next,)
        )

        NUTS_state = NUTS_state._replace(
            adapt_state=NUTS_state.adapt_state._replace(
                inverse_mass_matrix=state.inverse_mass_matrix,
                mass_matrix_sqrt_inv=inv_mass_flat_sqrt,
                mass_matrix_sqrt=mass_flat_sqrt
            )
        )
        self.first_adapt_state = NUTS_state.adapt_state
        self.NUTS_init = True
        return state._replace(
            rng_key=rng_key,
            SamplerState=NUTS_state
        )

    def divergent_update(self, state, new_step_size):
        # if only the step size and random key needs updating
        new_key = jax.random.split(state.rng_key, 1)[0]
        NUTS_state = state.SamplerState._replace(
            i=jnp.zeros_like(state.SamplerState.i) + self.num_warmup + 1,
            adapt_state=state.SamplerState.adapt_state._replace(
                step_size=jnp.ones_like(state.SamplerState.adapt_state.step_size) * new_step_size
            )
        )
        return state._replace(
            rng_key=new_key,
            SamplerState=NUTS_state
        )

    @partial(jax.jit, static_argnums=(0,))
    def vec_value_and_grad(self, beta, zs):
        return jax.vmap(
            jax.value_and_grad(self._potential_fn_gen(beta))
        )(zs)

    def replace_nuts(self, state):
        # Make a new NUTS state for a new beta value with a new
        # mass matrix
        inv_mass_flat_sqrt = jnp.sqrt(state.inverse_mass_matrix)
        mass_flat_sqrt = 1 / inv_mass_flat_sqrt
        step_size_estimate = jnp.exp(state.StepSizeState[1])
        adapt_state = self.first_adapt_state._replace(
            step_size=jnp.ones(self.num_particles) * step_size_estimate,
            inverse_mass_matrix=state.inverse_mass_matrix,
            mass_matrix_sqrt_inv=inv_mass_flat_sqrt,
            mass_matrix_sqrt=mass_flat_sqrt
        )
        rng_key_hmc, ke_key, rng_key = jax.random.split(state.rng_key, 3)
        KE = 0.5 * jax.random.chisquare(
            ke_key,
            df=state.inverse_mass_matrix.shape[1],
            shape=self.num_particles
        )
        PE, z_grad = self.vec_value_and_grad(state.beta_next, state.z_filter)
        zero_int = jnp.zeros(self.num_particles, dtype=jnp.result_type(int))
        zero_float = jnp.zeros(self.num_particles)
        zero_bool = jnp.zeros(self.num_particles, dtype=jnp.result_type(False))
        NUTS_state = HMCState(
            zero_int + self.num_warmup + 1,
            state.z_filter,
            z_grad,
            PE,
            PE + KE,
            None,
            None,
            zero_int,
            zero_float,
            zero_float,
            zero_bool,
            adapt_state,
            jax.random.split(rng_key_hmc, self.num_particles)
        )
        return state._replace(
            rng_key=rng_key,
            SamplerState=NUTS_state
        )

    def sample(self, state):
        step_size_estimate = self.first_step_size
        if not self.NUTS_init:
            # This is the first time NUTS has been run since class creation
            # make sure the sampler is created and initialized correctly
            sampler_kwargs = {
                'adapt_step': True,
                'step_size': step_size_estimate,
                'warmup': True
            }
            self.make_sampler(**sampler_kwargs)
            state = self.init_nuts(state)
        if state.count == 1:
            # this is the first draw, do a warmup with the correct
            # NUTS state
            draw_key, rng_key = jax.random.split(state.rng_key)
            cache_key = (
                numpyro.infer.mcmc._hashable(draw_key),
                numpyro.infer.mcmc._hashable(state.beta_next)
            )
            self.mcmc._init_state_cache[cache_key] = state.SamplerState
        else:
            # for any other draw use `post_warmup_state` to skip the warmup
            StepSizeState = state.StepSizeState
            step_size_estimate = jnp.exp(StepSizeState[1])
            state = self.replace_nuts(state)
            draw_key, rng_key = jax.random.split(state.rng_key)
            self.mcmc.post_warmup_state = state.SamplerState
        print(f'count {state.count}, beta {state.beta_next}, step size {step_size_estimate}')
        self.mcmc.run(draw_key, state.beta_next)

        if state.count == 1:
            step_size_estimate = self.mcmc._last_state.adapt_state.step_size.mean()
            # initialize step size state
            StepSizeState = self.ss_i(jnp.log(step_size_estimate))

        number_divergent = self.mcmc._states['diverging'].sum()
        if self.divergent_restart:
            while number_divergent > 0:
                step_size_estimate /= jnp.log2(2 + number_divergent)
                print(f'{number_divergent} divergent samples, trying again with smaller step size {step_size_estimate}')
                state = self.divergent_update(state, step_size_estimate)
                self.mcmc.post_warmup_state = state.SamplerState
                draw_key, rng_key = jax.random.split(state.rng_key)
                StepSizeState = self.ss_i(jnp.log(step_size_estimate))
                self.mcmc.run(draw_key, state.beta_next)
                number_divergent = self.mcmc._states['diverging'].sum()
        else:
            print(f'{number_divergent} divergent samples')
        z = self.mcmc.get_samples()
        log_likelihood = z['_log_likelihood']
        log_prior = z['_log_prior']
        z = {k: v for k, v in z.items() if not k.startswith('_log_')}
        # update step size
        target_accept_prob = self.inner_kernel._target_accept_prob
        current_accept_prob = self.mcmc._last_state.mean_accept_prob.mean()
        print(f'accept_prob: {current_accept_prob}')
        print(f'max_steps: {self.mcmc.last_state.num_steps.max()}')
        StepSizeState = self.ss_u(
            target_accept_prob - current_accept_prob,
            StepSizeState
        )
        # write new state
        state = state._replace(
            x=self.constrain(z),
            z=z,
            z_filter=None,
            rng_key=rng_key,
            beta=state.beta_next,
            log_prior=log_prior,
            log_likelihood=log_likelihood,
            SamplerState=None,
            StepSizeState=StepSizeState
        )
        print('Updating tempering')
        return self.update_smc(state)


def unconstrain_one(smc_instance, keys, z):
    z_filter = {k: z[k] for k in keys}
    return numpyro.infer.util.unconstrain_fn(
        smc_instance.model,
        smc_instance.model_args,
        smc_instance.model_kwargs,
        z_filter
    )


def get_smc_state_from_svi(smc_instance, multi_svi_results, guide, rng_key):
    draw_key, smc_key = jax.random.split(rng_key)
    guide_samples = guide.sample_posterior(
        draw_key,
        multi_svi_results.params,
        sample_shape=(smc_instance.num_samples,)
    )
    guide_samples_flat = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), guide_samples)

    prototype_params = numpyro.infer.util.find_valid_initial_params(
        rng_key,
        smc_instance.model,
        model_args=smc_instance.model_args,
        model_kwargs=smc_instance.model_kwargs,
        init_strategy=numpyro.infer.init_to_sample(),
        validate_grad=False
    )[0][0]
    keys = prototype_params.keys()
    guide_unconstrained_samples_flat = jax.lax.map(
        partial(unconstrain_one, smc_instance, keys), 
        guide_samples_flat
    )
    log_prior, log_likelihood = jax.lax.map(
        partial(
            split_prior_likelihood,
            smc_instance.model,
            smc_instance.model_args,
            smc_instance.model_kwargs
        ),
        guide_unconstrained_samples_flat
    )
    state = SMCTemperingState(
        count=0,
        x=guide_samples_flat,
        z=guide_unconstrained_samples_flat,
        z_filter=None,
        beta=1.0,
        beta_next=1.0,
        rng_key=smc_key,
        log_prior=log_prior,
        log_likelihood=log_likelihood,
        log_z_ratio=0.0,
        log_z_estimate=0.0,
        SamplerState=None,
        StepSizeState=None,
        inverse_mass_matrix=None
    )
    return smc_instance.update_smc(state)
