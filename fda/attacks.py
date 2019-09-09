import tensorflow as tf
import numpy as np
# import a few attack from cleverhans for easy retrieval in run_eval.py script.
from cleverhans.attacks.madry_et_al import MadryEtAl
from cleverhans.attacks.basic_iterative_method import BasicIterativeMethod
from cleverhans.attacks.momentum_iterative_method \
    import MomentumIterativeMethod
from cleverhans.attacks.attack import Attack
from cleverhans.model import Model, CallableModelWrapper

__all__ = ['MadryEtAl', 'BasicIterativeMethod', 'MomentumIterativeMethod']


class FDA(Attack):
    """
    Documentation
    """

    def __init__(self, model, sess=None, dtypestr='float32',
                 default_rand_init=True, **kwargs):
        """
        Based on ProjectedGradientDescent.
        """
        if not isinstance(model, Model):
            model = CallableModelWrapper(model, 'probs')

        super(FDA, self).__init__(model, sess=sess,
                                  dtypestr=dtypestr, **kwargs)
        self.feedable_kwargs = {
            'eps': self.np_dtype,
            'eps_iter': self.np_dtype,
            'y': self.np_dtype,
            'y_target': self.np_dtype,
            'clip_min': self.np_dtype,
            'clip_max': self.np_dtype
        }
        self.structural_kwargs = ['ord', 'nb_iter', 'rand_init']

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.

        :param x: The model's symbolic inputs.
        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        labels, nb_classes = self.get_or_guess_labels(x, kwargs)
        self.targeted = self.y_target is not None

        # Initialize loop variables
        adv_x = self.attack(x, labels)

        return adv_x

    def parse_params(self,
                     eps=0.3,
                     eps_iter=0.01,
                     nb_iter=40,
                     y=None,
                     ord=np.inf,
                     clip_min=None,
                     clip_max=None,
                     y_target=None,
                     rand_init=True,
                     **kwargs):
        """
        Take in a dictionary of parameters and applies attack-specific checks
        before saving them as attributes.

        Attack-specific parameters:

        :param eps: (required float) maximum distortion of adversarial example
                    compared to original input
        :param eps_iter: (required float) step size for each attack iteration
        :param nb_iter: (required int) Number of attack iterations.
        :param y: (optional) A tensor with the model labels.
        :param y_target: (optional) A tensor with the labels to target. Leave
                         y_target=None if y is also set. Labels should be
                         one-hot-encoded.
        :param ord: (optional) Order of the norm (mimics Numpy).
                    Possible values: np.inf, 1 or 2.
        :param clip_min: (optional float) Minimum input component value
        :param clip_max: (optional float) Maximum input component value
        :param rand_init: (optional bool) If True, an initial random
                    perturbation is added.
        """

        # Save attack-specific parameters
        self.eps = eps
        self.eps_iter = eps_iter
        self.nb_iter = nb_iter
        self.y = y
        self.y_target = y_target
        self.ord = ord
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.rand_init = rand_init

        if self.y is not None and self.y_target is not None:
            raise ValueError("Must not set both y and y_target")
        # Check if order of the norm is acceptable given current implementation
        if self.ord not in [np.inf, 1, 2]:
            raise ValueError("Norm order must be either np.inf, 1, or 2.")

        return True

    def get_opt_layers(self):
        opt_operations = []
        operations = tf.get_default_graph().get_operations()
        for op in operations:
            if 'while' in op.name:
                if 'resnet_v1_152' in op.name:
                    if op.type == u'Relu' and 'conv' not in op.name:
                        opt_operations.append(op.outputs)
                elif 'InceptionResnetV2' in op.name:
                    if op.type == u'Relu':
                        opt_operations.append(op.outputs)
                elif 'cell' in op.name:
                    if op.type == u'Relu':
                        opt_operations.append(op.outputs)
                else:
                    if op.type == u'Relu' and 'Mixed' not in op.name:
                        opt_operations.append(op.outputs)
                    if op.type == u'ConcatV2' and 'Mixed' in op.name:
                        opt_operations.append(op.outputs)
                if op.type == u'Mean' and 'resnet_v1_152/pool5' in op.name:
                    opt_operations.append(op.outputs)
                if op.type == u'AvgPool' and \
                        'InceptionV3/Logits/AvgPool_1a_8x8/AvgPool' in op.name:
                    opt_operations.append(op.outputs)
        return opt_operations

    def get_fda_loss(self, opt_operations):
        loss = 0
        for layer in opt_operations:
            layer = layer[0]
            batch_size = int(int(layer.shape[0]) / 2)
            tensor = layer[:batch_size]
            mean_tensor = tf.stack(
                [tf.reduce_mean(tensor, -1), ] * tensor.shape[-1], -1)
            wts_good = tensor < mean_tensor
            wts_good = tf.to_float(wts_good)
            wts_bad = tensor >= mean_tensor
            wts_bad = tf.to_float(wts_bad)
            loss += tf.log(tf.nn.l2_loss(
                wts_good * (layer[batch_size:]) / tf.cast(tf.size(layer),
                                                          tf.float32)))
            loss -= tf.log(tf.nn.l2_loss(
                wts_bad * (layer[batch_size:]) / tf.cast(tf.size(layer),
                                                         tf.float32)))
        loss = loss / len(opt_operations)
        return loss

    def loss_naive(self):
        # first we need to collect all the layers in the graph.
        opt_operations = self.get_opt_layers()

        print('Layers at which we optimize', opt_operations)
        # Now for each we need to get the loss:
        loss = self.get_fda_loss(opt_operations)
        return loss

    def attack_single_step(self, x, eta, y):
        """
        Given the original image and the perturbation computed so far, computes
        a new perturbation.

        :param x: A tensor with the original input.
        :param eta: A tensor the same shape as x that holds the perturbation.
        :param y: A tensor with the target labels or ground-truth labels.
        """
        from cleverhans.utils_tf import clip_eta

        adv_x = x + eta
        input_batch = tf.concat([x, adv_x], 0)
        logits = self.model.get_logits(input_batch)

        loss = self.loss_naive()
        grad, = tf.gradients(loss, adv_x)
        scaled_signed_grad = self.eps_iter * tf.sign(grad)
        adv_x = adv_x + scaled_signed_grad
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)
        eta = adv_x - x
        eta = clip_eta(eta, self.ord, self.eps)
        return eta

    def attack(self, x, y):
        """
        This method creates a symbolic graph that given an input image,
        first randomly perturbs the image. The
        perturbation is bounded to an epsilon ball. Then multiple steps of
        gradient descent is performed to increase the probability of a target
        label or decrease the probability of the ground-truth label.

        :param x: A tensor with the input image.
        """
        from cleverhans.utils_tf import clip_eta

        if self.rand_init:
            eta = tf.random_uniform(
                tf.shape(x), -self.eps, self.eps, dtype=self.tf_dtype)
            eta = clip_eta(eta, self.ord, self.eps)
        else:
            eta = tf.zeros_like(x)

        def cond(i, _):
            return tf.less(i, self.nb_iter)

        def body(i, e):
            new_eta = self.attack_single_step(x, e, y)
            return i + 1, new_eta

        _, eta = tf.while_loop(cond, body, [tf.zeros([]), eta], back_prop=True)

        adv_x = x + eta
        if self.clip_min is not None and self.clip_max is not None:
            adv_x = tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

        return adv_x
