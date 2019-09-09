import numpy as np

def init_metrics():
    metrics = {'normal_prediction': [],
               'adv_prediction': [],
               'gt_real': [],
               'old_label_rank_new': [],
               'new_label_rank_old': [],
               'real_acc': [],
               'adv_acc': [],
               'fr': []}
    return metrics


def update_metrics(metrics, normal_out, adv_out, gt_real, class_offset):
    # increased precision
    normal_out = normal_out.astype(np.float64)
    adv_out = adv_out.astype(np.float64)

    normal_top = np.argmax(normal_out, 1) - class_offset
    adv_top = np.argmax(adv_out, 1) - class_offset

    metrics['normal_prediction'].extend(normal_top)
    metrics['adv_prediction'].extend(adv_top)
    metrics['gt_real'].extend(gt_real)

    # the ranking based metrics
    normal_ranking = np.argsort(normal_out, 1)[:, ::-1] - class_offset
    adv_ranking = np.argsort(adv_out, 1)[:, ::-1] - class_offset
    # pdb.set_trace()
    old_label_rank_now = [list(adv_ranking[i]).index(normal_top[i]) for i in
                          range(normal_out.shape[0])]
    new_label_rank_old = [list(normal_ranking[i]).index(adv_top[i]) for i in
                          range(normal_out.shape[0])]
    metrics['old_label_rank_new'].extend(old_label_rank_now)
    metrics['new_label_rank_old'].extend(new_label_rank_old)

    # TODO: use incremental mean formulation rather than comparing the
    #  entire array at once.
    real_acc = np.mean(np.array(metrics['normal_prediction']) == np.array(
        metrics['gt_real']))
    adv_acc = np.mean(np.array(metrics['adv_prediction']) == np.array(
        metrics['gt_real']))
    fr = np.mean(np.array(metrics['adv_prediction']) != np.array(
        metrics['normal_prediction']))
    metrics['real_acc'].append(real_acc)
    metrics['adv_acc'].append(adv_acc)
    metrics['fr'].append(fr)

    return metrics
