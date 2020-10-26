import tensorflow as tf

# define the custom theta functions
@tf.custom_gradient
def theta_forget(x):
    def grad(dy):
        return dy * 0

    return tf.cast(x >= 0, tf.float32), grad


def get(surrogate, beta):
    if surrogate == "flat":

        @tf.custom_gradient
        def theta_flat(x):
            def grad(dy):
                return dy

            return tf.cast(x >= 0, tf.float32), grad

        return theta_flat

    if surrogate == "superspike":

        @tf.custom_gradient
        def theta_SuperSpike(x):
            def grad(dy):
                return dy * 1 / (beta * tf.abs(x) + 1) ** 2

            return tf.cast(x >= 0, tf.float32), grad
        return theta_SuperSpike

    if surrogate == "sigmoid":
        @tf.custom_gradient
        def theta_Sigmoid(x):
            def grad(dy):
                s = 1/(1+tf.exp(-beta*x))
                return dy * s * (1 - s)

            return tf.cast(x >= 0, tf.float32), grad

        return theta_Sigmoid

    if surrogate == "esser":
        @tf.custom_gradient
        def theta_Esser(x):
            def grad(dy):
                return dy * tf.max(0, 1-beta*tf.abs(x))

            return tf.cast(x >= 0, tf.float32), grad

        return theta_Esser