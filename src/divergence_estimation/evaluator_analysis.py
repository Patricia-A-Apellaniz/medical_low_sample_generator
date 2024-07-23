# This script evaluates the divergence between two distributions. It is used to evaluate the performance of the
# generative model.

import torch

from .evaluator import DivergenceEvaluator


def evaluate_set(x_real, x_gen, n, m, l, new_seed, results_path, pre_path=None, case='mvn', tsne_flag=False,
                 l_gt=50000, cfg=None, pr=None, ps=None, dataset_name=None):
    # This function is used to evaluate the divergence between x_real and x_gen given some parameters.

    # Define evaluator
    evaluator = DivergenceEvaluator(x_real, x_gen, id=new_seed, result_path=results_path,
                                    pre_path=pre_path, n=n, m=m, l=l, verbose=True, pr=pr, ps=ps,
                                    dataset_name=dataset_name)

    # Compute ground truth if distribution is known. Only for synthetic data.
    if case == 'mvn' or case == 'gm_dif' or case == 'gmm':
        kl_gt, js_gt = evaluator.compute_mc_gt(l_gt=l_gt, case=case)
        if kl_gt is not None and js_gt is not None:
            print(f"KL gt: {kl_gt}")
            print(f"JS gt: {js_gt}")
            evaluator.save_results(kl_gt, torch.tensor(0), torch.tensor(0), js_gt, torch.tensor(0), torch.tensor(0),
                                   results_path, case, m, l_gt, n, torch.tensor(0))

    kl1, kl2, kl3, js1, js2, js3 = None, None, None, None, None, None
    # Compute divergence
    if case in ['mvn', 'gm_dif', 'gmm']:  # Synthetic case.
        kl1, kl2, kl3, js1, js2, js3 = evaluator.evaluate(mc=True, ratio=False, disc=True, cfg=cfg)
    elif 'data' in case:  # Real data case.
        kl1, kl2, kl3, js1, js2, js3 = evaluator.evaluate(mc=False, ratio=False, disc=True, cfg=cfg)

    # Compute real KL divergence if distribution is known. Only for synthetic data.
    if case == 'mvn':
        kl_real = torch.distributions.kl_divergence(pr, ps)
        print(f"KL real: {kl_real}")
    else:
        kl_real = torch.tensor(0)
    # Save results
    evaluator.save_results(kl1, kl2, kl3, js1, js2, js3, results_path, case, m, l, n, new_seed, kl_real=kl_real)
