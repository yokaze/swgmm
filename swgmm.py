import numpy as np
import subprocess
import tensorflow as tf
from matplotlib import pyplot as pl

# Retrieve Old Faithful geyser data used in PRML.
# This dataset is somewhat difficult to obtain: this time I read it from R.
def get_faithful_data():
    text = subprocess.check_output(['r', '-q', '-e', 'faithful'])
    text = text.decode('ascii')
    text = text.splitlines()
    ret = []
    for line in text:
        values = line.split()
        if (len(values) == 3):
            x = float(values[1])
            y = float(values[2])
            ret.append([x, y])
    return np.array(ret, dtype=np.float64)

# Normalize dataset to enhance optimizer.
faithful = get_faithful_data()
faithful -= np.mean(faithful, axis=0)[None, :]
faithful /= np.std(faithful, axis=0)[None, :]

# The following lines render Old Faithful dataset and 2D Gaussian distributions.
# mu represents mean, and var represents standard deviations for x and y axes.
def draw_ring(mu, dev, alpha=1):
    angles = np.linspace(0, 2 * np.pi, 100)
    x = mu[0] + dev[0] * np.cos(angles)
    y = mu[1] + dev[1] * np.sin(angles)
    pl.plot(x, y, 'b-', alpha=alpha)

def draw_directions(directions):
    for i in range(len(directions)):
        d = directions[i]
        pl.plot([d[0] * -5, d[0] * 5], [d[1] * -5, d[1] * 5], 'b-', alpha=.125)

def draw(pi, mu, var, directions):
    draw_directions(directions)
    pl.plot(faithful[:, 0], faithful[:, 1], 'b+', alpha=.5)
    pl.plot(mu[:, 0], mu[:, 1], 'go')
    for i in range(len(mu)):
        draw_ring(mu[i], np.sqrt(var[i]))
        draw_ring(mu[i], np.sqrt(pi[i] / pi.mean()) * np.sqrt(var[i]), alpha=.25)
    pl.xlim(-2.5, 2.5)
    pl.ylim(-2.5, 2.5)

# Calculate variants of Gaussian integrals.
def integrate_emx2(a, b):
    return .5 * np.sqrt(np.pi) * (tf.math.erf(b) - tf.math.erf(a))

def integrate_xemx2(a, b):
    return .5 * (tf.exp(-a * a) - tf.exp(-b * b))

def integrate_x2emx2(a, b):
    A = .25 * np.sqrt(np.pi) * tf.math.erf(a) - .5 * a * tf.exp(-a * a)
    B = .25 * np.sqrt(np.pi) * tf.math.erf(b) - .5 * b * tf.exp(-b * b)
    return B - A

# Execute M step of EM algorithm.
# This method is just used to obtain initial parameters for SWGMM.
def calc_mstep(z):
    pi = z.sum(0) / z.sum()
    mu = (z[:, :, None] * faithful[:, None, :]).sum(0) / z.sum(0)[:, None]
    var = (z[:, :, None] * ((faithful[:, None, :] - mu[None, :, :]) ** 2)).sum(0)
    var /= z.sum(0)[:, None]
    return pi, mu, var

# Initialize responsibility using uniform random variables and make initial values for latent variables.
# This configuration forces the estimator to consume a large number of iterations:
# it thus makes easier to watch the character of learning algorithms.
def sample_init_parameter():
    z = np.random.dirichlet(np.ones(nclass), len(faithful))
    pi, mu, var = calc_mstep(z)
    lpi = tf.Variable(np.log(pi))
    mu = tf.Variable(mu)
    lv = tf.Variable(np.log(var))
    return lpi, mu, lv

# Sample direction vector for Radon transform.
def sample_direction():
    angle = 2 * np.pi * np.random.rand()
    return tf.constant([np.cos(angle), np.sin(angle)])

# Sample a set of directions, that used for approximating sliced Wasserstein distance.
# If fixed parameter is filled with vectors, some of the return values are fixed with them.
def sample_directions(ndirection, fixed=None):
    if (fixed is not None):
        assert len(fixed) <= ndirection
        ret = list(fixed)
    else:
        ret = []

    while (len(ret) < ndirection):
        ret.append(sample_direction())
    ret = np.array(ret)
    return tf.constant(ret, dtype=tf.float64)

# Compute projected coordinations of vectors using inner-product with direction vector.
def project_vector(x, direction):
    if (x.shape.rank == 1):
        x = x[None, :]
    ret = tf.reduce_sum(x * direction[None, :], axis=1)
    return tf.squeeze(ret)

# Compute projected variance of diagonal matrix diag(V).
def project_variance(var, direction):
    if (var.shape.rank == 1):
        var = var[None, :]
    ret = tf.reduce_sum(var * (direction * direction)[None, :], axis=1)
    return tf.squeeze(ret)

# Calculate PDF of one-dimensional Gaussian distribution.
def gaussian_pdf(x, mu, var):
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    mu = tf.convert_to_tensor(mu, dtype=tf.float64)
    prec = 1. / tf.convert_to_tensor(var, dtype=tf.float64)
    if (x.shape.rank == 0):
        x = x[None]
    if (mu.shape.rank == 0):
        mu = mu[None]
        prec = prec[None]
    ret = tf.sqrt(.5 * prec[None, :] / np.pi)
    ret *= tf.exp(-.5 * prec[None, :] * (x[:, None] - mu[None, :]) * (x[:, None] - mu[None, :]))
    return tf.squeeze(ret)

# Calculate CDF of one-dimensional Gaussian distribution.
def gaussian_cdf(x, mu, var):
    x = tf.convert_to_tensor(x, dtype=tf.float64)
    mu = tf.convert_to_tensor(mu, dtype=tf.float64)
    var = tf.convert_to_tensor(var, dtype=tf.float64)
    if (x.shape.rank == 0):
        x = x[None]
    if (mu.shape.rank == 0):
        mu = mu[None]
    ret = .5 * (1 + tf.math.erf((x[:, None] - mu[None, :]) / tf.sqrt(2 * var)))
    return tf.squeeze(ret)

# Calculate PDF of one-dimensional Gaussian mixture distribution.
def gaussian_mixture_pdf(x, pi, mu, var):
    pi = tf.convert_to_tensor(pi, dtype=tf.float64)
    return tf.reduce_sum(pi[None, :] * gaussian_pdf(x, mu, var), axis=1)

# Calculate CDF of one-dimensional Gaussian mixture distribution.
def gaussian_mixture_cdf(x, pi, mu, var):
    pi = tf.convert_to_tensor(pi, dtype=tf.float64)
    return tf.reduce_sum(pi[None, :] * gaussian_cdf(x, mu, var), axis=1)

# Calculate inverse of CDF of one-dimensional Gaussian mixture distribution.
# This function uses binary search to compute the value, utilizing that the function is monotonic.
# This function provides custom gradient for the sake of automatic differentiation.
@tf.custom_gradient
def gaussian_mixture_cdfinv(r, pi, mu, var):
    r = tf.convert_to_tensor(r, dtype=tf.float64)
    pi = tf.convert_to_tensor(pi, dtype=tf.float64)
    mu = tf.convert_to_tensor(mu, dtype=tf.float64)
    var = tf.convert_to_tensor(var, dtype=tf.float64)
    if (r.shape.rank == 0):
        r = r[None]
    xmin = -1
    xmax = 1
    while (tf.reduce_sum(pi * gaussian_cdf(xmin, mu, var)) > r[0]):
        xmin *= 2
    while (tf.reduce_sum(pi * gaussian_cdf(xmax, mu, var)) < r[-1]):
        xmax *= 2
    xmin = tf.tile(tf.convert_to_tensor([xmin], dtype=tf.float64), r.shape)
    xmax = tf.tile(tf.convert_to_tensor([xmax], dtype=tf.float64), r.shape)
    for i in range(50):
        xmid = (xmin + xmax) * .5
        cur_ratio = tf.reduce_sum(pi[None, :] * gaussian_cdf(xmid, mu, var), axis=1)
        mask = tf.cast(r < cur_ratio, tf.float64)
        xmin = xmin * mask + xmid * (1 - mask)
        xmax = xmid * mask + xmax * (1 - mask)
    ret = (xmin + xmax) * .5

    def grad(_dx):
        gpdf = gaussian_pdf(ret, mu, var)
        gmpdf = gaussian_mixture_pdf(ret, pi, mu, var)
        _dr = _dx / gaussian_mixture_pdf(ret, pi, mu, var)
        _dpi = -tf.reduce_sum(_dx[:, None] * gaussian_cdf(ret, mu, var) / gmpdf[:, None], axis=0)
        _dmu = tf.reduce_sum(_dx[:, None] * pi[None, :] * gpdf / gmpdf[:, None], axis=0)
        _dvar = tf.reduce_sum(_dx[:, None] * pi[None, :] / (2 * var[None, :]) * (ret[:, None] - mu[None, :]) * gpdf / gmpdf[:, None], axis=0)
        return [_dr, _dpi, _dmu, _dvar]
    return ret, grad

# Compute Wasserstein distance between observed data X and a one-dimensional Gaussian mixture distribution.
# It implements 1-Wasserstein and 2-Wasserstein only.
def gaussian_mixture_wasserstein_loss(x, pi, mu, var, order):
    # Sort input data for computing alignment with Gaussian mixture.
    x = tf.sort(x)
    nx = x.shape[0]

    # Calculate variances of the distributions.
    prec = 1. / var

    # Split Gaussian mixture distribution into N parts to compute transportation cost.
    # It also computes the split point between right-facing transporation and left-facing transportation
    # to compute 1-Wasserstein integral properly.
    ratio = tf.cast(tf.linspace(1. / nx, 1 - 1. / nx, nx - 1), tf.float64)
    partition = gaussian_mixture_cdfinv(ratio, pi, mu, var)
    partition_left = tf.concat([[-1e+10], partition], axis=0)
    partition_right = tf.concat([partition, [1e+10]], axis=0)
    partition_mid = tf.minimum(tf.maximum(partition_left, x), partition_right)

    # Change of variables, for later integrals
    integral_left = (partition_left[:, None] - mu[None, :]) * tf.sqrt(.5 * prec)[None, :]
    integral_mid = (partition_mid[:, None] - mu[None, :]) * tf.sqrt(.5 * prec)[None, :]
    integral_right = (partition_right[:, None] - mu[None, :]) * tf.sqrt(.5 * prec)[None, :]

    if (order == 1):
        loss_left = (x[:, None] - mu[None, :]) * integrate_emx2(integral_left, integral_mid)
        loss_left -= 1. / tf.sqrt(.5 * prec[None, :]) * integrate_xemx2(integral_left, integral_mid)
        loss_left *= tf.cast(1. / tf.sqrt(np.pi), tf.float64)
        loss_right = 1. / tf.sqrt(.5 * prec[None, :]) * integrate_xemx2(integral_mid, integral_right)
        loss_right -= (x[:, None] - mu[None, :]) * integrate_emx2(integral_mid, integral_right)
        loss_right *= tf.cast(1. / tf.sqrt(np.pi), tf.float64)
        return pi[None, :] * (loss_left + loss_right)
    elif (order == 2):
        diff = x[:, None] - mu[None, :]
        loss = (diff * diff) * integrate_emx2(integral_left, integral_right)
        loss -= 2 * diff / tf.sqrt(.5 * prec[None, :]) * integrate_xemx2(integral_left, integral_right)
        loss += 1. / (.5 * prec[None, :]) * integrate_x2emx2(integral_left, integral_right)
        loss *= tf.cast(1. / tf.sqrt(np.pi), tf.float64)
        return pi[None, :] * loss
    else:
        assert False

def estimate(nstep, ndirection, fixed_directions=None, order=2, use_adam=True):
    faith = tf.constant(faithful)
    lpi, mu, lv = sample_init_parameter()

    if use_adam:
        opt = tf.keras.optimizers.Adam(learning_rate=.2)
    else:
        opt = tf.keras.optimizers.RMSprop(learning_rate=.05, centered=True)

    loss_history = []
    for istep in range(nstep):
        directions = sample_directions(ndirection, fixed_directions)

        # Compute approximate sliced Wasserstein distance between empirical and model distributions.
        def sw_loss():
            total_loss = 0
            for idirection in range(ndirection):
                direction = directions[idirection]
                faith_proj = project_vector(faith, direction)
                lpi_normal = lpi - tf.reduce_logsumexp(lpi)
                pi = tf.exp(lpi_normal)
                mu_proj = project_vector(mu, direction)
                var_proj = project_variance(tf.exp(lv), direction)
                projected_loss = gaussian_mixture_wasserstein_loss(faith_proj, pi, mu_proj, var_proj, order)
                total_loss += tf.reduce_sum(projected_loss)
            return total_loss / ndirection

        # Render inference situation to graphs.
        # Inference status (left-top), Wasserstein distance along x-axis (left-middle),
        # alignment of empirical and estimated CDFs (left-bottom),
        # and the computed approximate sliced Wasserstein distance (right).
        def draw_figure():
            pl.clf()

            pl.subplot(321)
            lpi_normal = lpi - tf.reduce_logsumexp(lpi)
            pi = tf.exp(lpi_normal)
            draw(pi.numpy(), mu.numpy(), np.exp(lv.numpy()), directions.numpy())

            pl.subplot(323)
            direction = tf.convert_to_tensor([1, 0], dtype=tf.float64)
            faith_proj = tf.sort(project_vector(faith, direction))
            mu_proj = project_vector(mu, direction)
            var_proj = project_variance(tf.exp(lv), direction)
            loss = gaussian_mixture_wasserstein_loss(faith_proj, pi, mu_proj, var_proj, order=1)
            loss = tf.reduce_sum(loss, axis=1)
            pl.plot(faith_proj, loss, 'b+', alpha=.5)
            pl.xlim(-2.5, 2.5)
            pl.ylim(0, 0.01)

            pl.subplot(325)
            nx = faith_proj.shape[0]
            ratio = tf.cast(tf.linspace(1. / (2 * nx), (2 * nx - 1) / (2 * nx), nx), tf.float64)
            p = gaussian_mixture_cdfinv(ratio, pi, mu_proj, var_proj)
            pl.plot(faith_proj, ratio)
            pl.plot(p, ratio)
            pl.xlim(-2.5, 2.5)
            pl.ylim(0, 1)

            pl.subplot(122)
            pl.plot(loss_history)
            pl.xlim(0, nstep)
            pl.ylim(0, (loss_history[0] * 1.2).numpy())
            pl.tight_layout()

        # Update variables.
        opt.minimize(sw_loss, var_list=[lpi, mu, lv])

        # Compute current loss.
        loss_history.append(sw_loss() ** (1. / order))

        # Render graphs per iteration.
        # Render twice on the first iteration because tight_layout runs glitchy on my environment.
        # (This behavior is observed on my environment, macOS Mojave + Python3,
        #  this code can be safely removed if nothing will happen on your machine)
        draw_figure()
        if (istep == 0):
            draw_figure()
        pl.pause(.1)

if (__name__ == '__main__'):
    # Set graph size.
    pl.figure(figsize=[7.5, 5])

    # Set the number of Gaussian components and the number of iterations.
    nclass = 5
    nstep = 100

    # Set the number of directions used for sliced Wasserstein distance approximation.
    # The calculation converges to the exact SW distance when ndirection = infty.
    ndirection = 5

    # If you would like to fix some of the direction vectors, please put it here.
    fixed_directions = None
    # fixed_directions = [[1, 0]]
    # fixed_directions = [[1, 0], [0, 1]]
    # fixed_directions = [[1., 0.], [0., 1.], [np.sqrt(2), np.sqrt(2)]]
    # fixed_directions = [[1., 0.], [0., 1.], [np.sqrt(2), np.sqrt(2)], [np.sqrt(2), -np.sqrt(2)]]

    # Set the order of Wasserstein distance.
    # This implementation allows one and two only.
    order = 2

    # Determine to use Adam or RMSprop.
    use_adam = True
    estimate(nstep, ndirection, fixed_directions, order, use_adam)

    # Halt the program when inference is done.
    pl.show()
