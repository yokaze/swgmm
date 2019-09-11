from swgmm import *

x = np.random.randn(400)
x[:100] *= .5
x[:100] -= 8
y = np.random.randn(400)
y[:100] *= .5

pl.figure(figsize=[7.5, 5])
pl.subplot(311)
pl.plot(x, y, 'b+')
pl.xlim(-21, 5)
pl.ylim(-2.5, 2.5)
draw_ring([-15, 0], [1, 1])
draw_ring([-4, 0], [4, 1])
pl.plot(-15, 0, 'go')
pl.plot(-4, 0, 'go')

x = tf.cast(tf.constant(x), tf.float64)
pi = tf.cast(tf.constant([0.5, 0.5]), tf.float64)
var = tf.cast(tf.constant([1, 4]), tf.float64)

xlist = np.linspace(-21, 5, 101)
pl.subplot(312)
kl_loss = []
for m in xlist:
    mu = tf.cast(tf.constant([m, -4]), tf.float64)
    lp = np.log(gaussian_mixture_pdf(x, pi, mu, var).numpy())
    kl_loss.append(-lp.sum())
pl.plot(xlist, kl_loss)
pl.legend(['KL'], loc='lower left')
pl.xlim(-21, 5)

pl.subplot(313)
w1_loss = []
w2_loss = []
for m in xlist:
    mu = tf.cast(tf.constant([m, -4]), tf.float64)
    w1_loss.append(tf.reduce_sum(gaussian_mixture_wasserstein_loss(x, pi, mu, var, order=1)).numpy())
    w2_loss.append(np.sqrt(tf.reduce_sum(gaussian_mixture_wasserstein_loss(x, pi, mu, var, order=2)).numpy()))
pl.plot(xlist, w1_loss)
pl.plot(xlist, w2_loss)
pl.legend(['W1', 'W2'], loc='lower left')
pl.xlim(-21, 5)

pl.tight_layout()
pl.show()
