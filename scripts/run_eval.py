from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml
import logging
import argparse
import numpy as np

import tensorflow as tf
# TF-SLIM imports
from nets import nets_factory
from preprocessing import preprocessing_factory
from cleverhans import model as chm

# from FDA
from fda.data_loader import DataLoader
from fda import utils
from fda import attacks
from fda import metrics as metrics_functions


def config_specific_params(config):
    configured_params = config
    attack_config = configured_params['attack_config']
    attack_params = configured_params['attack_params']
    graph_config = configured_params['graph_config']
    configured_params['experiment_name'] = \
        '_'.join([graph_config['model'], attack_config['attack_name'],
                  str(attack_params['eps']), str(attack_params['nb_iter']),
                  str(attack_params['eps_iter'])])
    configured_params['log_file'] = \
        os.path.join('results', configured_params['experiment_name'] + '.log')
    configured_params['results_file'] = \
        os.path.join('results', configured_params['experiment_name'] + '.txt')

    # based on model input ratio set the attack size.
    if configured_params['data_config']['normalize']:
        attack_params['eps'] = attack_params['eps']/128.
        attack_params['eps_iter'] = attack_params['eps_iter']/128.
        attack_params['clip_min'], attack_params['clip_max'] = (-1.0, 1.0)
    else:
        attack_params['clip_min'], attack_params['clip_max'] = (-124.0, 153.0)
    attack_params['batch_size'] = \
        configured_params['data_config']['batch_size']
    
    graph_config['batch_size'] = \
        configured_params['data_config']['batch_size']
    graph_config['offset'] = graph_config['num_classes'] - 1000

    configured_params['attack_params'] = attack_params
    configured_params['graph_config'] = graph_config

    return configured_params


def run_eval(params):

    # assign names to the different params:
    graph_config = params['graph_config']
    data_config = params['data_config']
    attack_config = params['attack_config']
    attack_params = params['attack_params']
    batch_size = data_config['batch_size']
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

    session = tf.Session(config=gpu_config)

    # fetch the logger
    logger = logging.getLogger('logger')

    # setup the session and the graphs:
    network_fn = nets_factory.get_network_fn(
        graph_config['model'],
        num_classes=(graph_config['num_classes']),
        is_training=False)

    # set up inputs
    eval_image_size = network_fn.default_image_size
    preprocessing_name = graph_config['model']
    image_preprocessing_fn = \
        preprocessing_factory.get_preprocessing(preprocessing_name,
                                                is_training=False)

    data_loader = DataLoader(data_config,
                             image_preprocessing_fn, eval_image_size)

    # setup model
    logits, normal_network = network_fn(data_loader.input_images)

    net_varlist = [v for v in tf.get_collection(tf.GraphKeys.VARIABLES)]
    saver = tf.train.Saver(var_list=net_varlist)
    saver.restore(session, graph_config['checkpoint_file'])

    # also setup evaluation network.
    input_placeholder = \
        tf.placeholder(shape=[None, eval_image_size, eval_image_size, 3],
                       dtype=tf.float32, name='Input_placeholder')
    logits_frm_placeholder, network_frm_placeholder = \
        network_fn(input_placeholder, reuse=tf.AUTO_REUSE)

    logger.info('Loaded the networks')

    # setup the cleverhans attack:
    clever_model = chm.CallableModelWrapper(network_fn, 'logits')
    attacker_class = getattr(attacks, attack_config['attack_name'])
    attacker = attacker_class(clever_model, sess=session)

    # Output
    if attack_config['attack_type'] == "GT":
        outputs = tf.placeholder(shape=[None, graph_config['num_classes']],
                                 dtype=tf.float32, name='GT_placeholder')
        adversarial_x_tensor = \
            attacker.generate(data_loader.input_images, y=outputs,
                              ** attack_params)
    elif attack_config['attack_type'] == "self":
        adversarial_x_tensor = \
            attacker.generate(data_loader.input_images,
                              ** attack_params)

    logger.info('Loaded the attack')

    # Evaluation
    im_list = open(data_config['image_list']).readlines()[::5]
    gt_list = open(data_config['gt_list']).readlines()[::5]
    im_list = [x.strip() for x in im_list]
    gt_list = [x.strip() for x in gt_list]

    total_iter = len(im_list) / batch_size

    # metrics
    metrics = metrics_functions.init_metrics()

    # define stylize net

    for i in range(int(total_iter)):
        if i % 10 == 0:
            logger.info('iter' + str(i) + ' of ' + str(total_iter))

        im_batch = im_list[i * batch_size:
                           (i + 1) * batch_size]
        gt_real = gt_list[i * batch_size:
                          (i + 1) * batch_size]
        gt_real = [int(x) for x in gt_real]
        gt_batch = utils.get_gt_batch(gt_real, graph_config)

        if attack_config['attack_type'] == "GT":
            feed_dict = {data_loader.image_path: im_batch, outputs: gt_batch}
        elif attack_config['attack_type'] == "self":
            feed_dict = {data_loader.image_path: im_batch}
        else:
            raise ValueError("attack type has to be GT or self")

        # find the adversarial sample
        adversarial_x = session.run(adversarial_x_tensor, feed_dict=feed_dict)

        # perform evaluation
        feed_dict = {data_loader.image_path: im_batch,
                     input_placeholder: adversarial_x}
        normal_out, adv_out = \
            session.run([logits, logits_frm_placeholder], feed_dict=feed_dict)

        metrics = metrics_functions.update_metrics(metrics, normal_out,
                                                   adv_out, gt_real,
                                                   graph_config['offset'])

        normal_out = metrics['normal_prediction'][-1]
        adv_out = metrics['adv_prediction'][-1]

        log_txt = ' '.join(['Adversarial prediction', str(adv_out),
                            'true_predictions', str(normal_out),
                            'true_gt ', str(gt_real[-1])])
        logger.info(log_txt)

        logger.info('======== ITER ' + str(i) + "========")
        logger.info('Real Top-1 Accuracy = {:.2f}'.format(
            metrics['real_acc'][-1]))
        logger.info('Top-1 Accuracy = {:.2f}'.format(
            metrics['adv_acc'][-1]))
        logger.info('Fooling Rate = {:.2f}'.format(
            metrics['fr'][-1]))
        logger.info('Old Label New Rank = {:.2f}'.format(
            np.mean(metrics['old_label_rank_new'])))
        logger.info('New Label Old Rank = {:.2f}'.format(
            np.mean(metrics['new_label_rank_old'])))

        # to be on the safe side, log the details of file
        if i % 10 == 0:
            normal_x = session.run(data_loader.input_images, feed_dict)
            pert = normal_x - adversarial_x
            log_txt = ' '.join(['Image min max', str(np.max(normal_x)),
                               str(np.min(normal_x))])
            logger.info(log_txt)
            log_txt = ' '.join(['Perturbation min max', str(np.max(pert)),
                               str(np.min(pert))])
            logger.info(log_txt)

    # now save all the important details.
    result = open(params['results_file'], 'w')
    result.write('Real Top-1 Accuracy = {:.4f}'.format(
        metrics['real_acc'][-1]) + '\n')
    result.write('Top-1 Accuracy = {:.4f}'.format(
        metrics['adv_acc'][-1]) + '\n')
    result.write('Fooling Rate = {:.4f}'.format(
        metrics['fr'][-1]) + '\n')
    result.write('Old Label New Rank = {:.4f}'.format(
        np.mean(metrics['old_label_rank_new'])) + '\n')
    result.write('New Label Old Rank = {:.4f}'.format(
        np.mean(metrics['new_label_rank_old'])) + '\n')
    result.close()


if __name__ == '__main__':

    # Parser config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        default='config/vgg_config.yaml',
                        help='The configuration file. Check '
                             'config/test_config.yaml for more information')

    args = parser.parse_args()
    config = yaml.load(open(args.config_file))
    eval_params = config_specific_params(config)

    # setup the logger.
    _ = utils.setup_logger('logger', eval_params['log_file'])

    run_eval(eval_params)
