from swgmm import *

np.random.seed(6)

x = tf.cast(np.random.randn(10) * 4 + 1.5, tf.float64)
x = tf.sort(x)
pi = tf.cast(tf.constant([0.5, 0.5]), tf.float64)
mu = tf.cast(tf.constant([-1, 4]), tf.float64)
var = tf.cast(tf.constant([4, 1]), tf.float64)

pl.figure(figsize=[6, 4])

xlist = np.linspace(-20, 20, 1001)
gmpdf = gaussian_mixture_pdf(xlist, pi, mu, var).numpy()

partition = gaussian_mixture_cdfinv(np.linspace(0.05, 0.95, 19), pi, mu, var)
partition = np.hstack([-10, partition.numpy(), 10])

pl.plot([-20, 20], [0, 0], color='0.75')
for i in range(10):
    pl.plot([x[i], x[i]], [0, .2], color='0.875', linestyle='dashed')
    pl.plot([partition[i * 2 + 1], x[i]], [0, .2], color='.75')

for i in range(9):
    px = partition[i * 2 + 2]
    py = np.interp(px, xlist, gmpdf)
    pl.plot([px, px], [0, -py], color='0.75')
pl.plot(x, np.ones_like(x) * .2, 'go')
pl.plot(xlist, -gmpdf)
pl.xlim(-10, 10)
pl.xticks([])
pl.yticks([])
pl.tight_layout()
pl.show()
