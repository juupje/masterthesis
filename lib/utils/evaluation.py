import numpy as np

def evaluate_sr_vs_cr(y_pred_cr, y_true_cr, y_pred_sr, y_true_sr, epsB=1e-3, Rsys=0, sigma_sys=0, quiet=False):
    to_select = int(np.round(y_pred_cr.shape[0]*epsB))
    thres = np.partition(y_pred_cr, -to_select)[-to_select]
    is_selected_cr = y_pred_cr>=thres
    is_background_cr = y_true_cr==0
    n_selected_bg_in_cr = np.sum(is_selected_cr&is_background_cr)
    n_selected_sn_in_cr = np.sum(is_selected_cr& ~is_background_cr)
    n_selected_cr       = np.sum(is_selected_cr)
    n_cr = y_pred_cr.shape[0]

    is_selected_sr = y_pred_sr>=thres
    is_background_sr = y_true_sr==0
    n_selected_bg_in_sr = np.sum(is_selected_sr&is_background_sr)
    n_selected_sn_in_sr = np.sum(is_selected_sr& ~is_background_sr)
    n_selected_sr = np.sum(is_selected_sr)
    n_sr = y_pred_sr.shape[0]
    
    N_exp = epsB*n_sr*(1+Rsys)
    sigma_exp_stat = 1/np.sqrt(n_selected_cr)
    sigma_exp = N_exp*np.sqrt(sigma_exp_stat**2+sigma_sys**2)
    sigma_stat = np.sqrt(1/N_exp+1/n_selected_cr)
    #Asimov Estimate
    S = np.sqrt(2*(n_selected_sr*np.log(n_selected_sr*(N_exp+sigma_exp**2)/(N_exp**2+n_selected_sr*sigma_exp**2))
                   -N_exp**2/(sigma_exp**2)*np.log1p((sigma_exp**2)*(n_selected_sr-N_exp)/(N_exp*(N_exp+sigma_exp**2)))))
    sigma_simple = np.sqrt(N_exp+sigma_exp**2)
    S_simple = (n_selected_sr-N_exp)/sigma_simple
    sigma = np.sqrt(n_selected_cr*(1+n_sr/n_cr))
    alpha = (n_selected_sr-N_exp)/sigma
    if not quiet:
        print(f"At eps_b={epsB:f}%, selected:")
        print("\tIn CR:")
        print(f"\t\t{n_selected_bg_in_cr:d} background events of {np.sum(is_background_cr):d}")
        print(f"\t\t{n_selected_sn_in_cr:d} signal events of {np.sum(~is_background_cr):d}")
        print("\tIn SR:")
        print(f"\t\t{np.sum(is_selected_sr):d} events selected in total")
        print(f"\t\t{n_selected_bg_in_sr:d} background events of {np.sum(is_background_sr):d}")
        print(f"\t\t{n_selected_sn_in_sr:d} signal events of {np.sum(~is_background_sr):d}")
        print(f"\t\tExpected events in SR: {N_exp}")
        print(f"\tsigma_sys: {sigma_sys:.3f}, sigma_stat: {sigma_stat:.3f}")
        print(f"\tsigma_exp_stat: {sigma_exp_stat:.3f}, sigma_exp: {sigma_exp:.3f}")
        print(f"\tS-measure: {S:.3f}")
        print(f"\tsigma_simple: {sigma_simple:.3f}")
        print(f"\tS-simple: {S_simple:.3f}")
        print(f"\tsigma_old: {sigma:.3f}")
        print(f"\tSignificance: {alpha:.3f}")
    return {"eps_b": epsB,
            "cr": {"background": np.sum(is_background_cr), "signal": np.sum(~is_background_cr),
                    "selected": {"total": np.sum(n_selected_cr), "background": np.sum(n_selected_bg_in_cr), "signal": np.sum(n_selected_sn_in_cr)}},
            "sr": {"background": np.sum(is_background_sr), "signal": np.sum(~is_background_sr),
                    "selected": {"total": np.sum(n_selected_sr), "background": np.sum(n_selected_bg_in_sr), "signal": np.sum(n_selected_sn_in_sr)},
                    "expected": N_exp},
            "sigma": {"stat": sigma_stat, "sys": sigma_sys, "exp_stat": sigma_exp_stat, "exp": sigma_exp, "simple": sigma_simple, "old": sigma},
            "S": S, "S-simple": S_simple, "significance": alpha}

def evaluate_sr_vs_cr_kfold(y_pred_cr:list, y_true_cr:list, y_pred_sr:list, y_true_sr:list, epsB=1e-3, Rsys=0, sigma_sys=0, quiet=False):
    folds = range(len(y_pred_cr))
    result = {"folds": []}
    n_selected_bg_in_cr, n_selected_sn_in_cr = [], []
    n_selected_bg_in_sr, n_selected_sn_in_sr = [], []
    n_selected_sr, n_selected_cr, n_sr, n_cr = [], [], [], []
    for fold in folds:
        to_select = int(np.round(y_pred_cr[fold].shape[0]*epsB))
        thres = np.partition(y_pred_cr[fold], -to_select)[-to_select]
        is_selected_cr = y_pred_cr[fold]>=thres
        is_background_cr = y_true_cr[fold]==0
        n_selected_bg_in_cr.append(np.sum(is_selected_cr&is_background_cr))
        n_selected_sn_in_cr.append(np.sum(is_selected_cr& ~is_background_cr))
        n_selected_cr.append(np.sum(is_selected_cr))
        n_cr.append(y_pred_cr[fold].shape[0])

        is_selected_sr = y_pred_sr[fold]>=thres
        is_background_sr = y_true_sr[fold]==0
        n_selected_bg_in_sr.append(np.sum(is_selected_sr&is_background_sr))
        n_selected_sn_in_sr.append(np.sum(is_selected_sr& ~is_background_sr))
        n_selected_sr.append(np.sum(is_selected_sr))
        n_sr.append(y_pred_sr[fold].shape[0])
        result["folds"].append({"cr": {"background": np.sum(is_background_cr), "signal": np.sum(~is_background_cr),
                                    "selected": {"total": np.sum(is_selected_cr), "background": n_selected_bg_in_cr[-1], "signal": n_selected_sn_in_cr[-1]}},
                                "sr": {"background": np.sum(is_background_sr), "signal": np.sum(~is_background_sr),
                                        "selected": {"total": np.sum(is_selected_sr), "background": n_selected_bg_in_sr[-1], "signal": n_selected_sn_in_sr[-1]}}})
    total_bg_in_cr = np.sum([np.sum(y_true_cr[fold]==0) for fold in folds])
    total_sn_in_cr = np.sum([np.sum(y_true_cr[fold]==1) for fold in folds])
    total_bg_in_sr = np.sum([np.sum(y_true_sr[fold]==0) for fold in folds])
    total_sn_in_sr = np.sum([np.sum(y_true_sr[fold]==1) for fold in folds])

    total_selected_sr = np.sum(n_selected_sr)
    total_selected_cr = np.sum(n_selected_cr)
    if not quiet:
        print(f"Combining {len(y_pred_cr)} folds")
        print(f"At eps_b={epsB:f}%, selected:")
        print("\tIn CR:")
        print(f"\t\t{np.sum(n_selected_bg_in_cr):d} background events of {total_bg_in_cr:d}")
        print(f"\t\t{np.sum(n_selected_sn_in_cr):d} signal events of {total_sn_in_cr:d}")
        print("\tIn SR:")
        print(f"\t\t{np.sum(n_selected_sr):d} events selected in total")
        print(f"\t\t{np.sum(n_selected_bg_in_sr):d} background events of {total_bg_in_sr:d}")
        print(f"\t\t{np.sum(n_selected_sn_in_sr):d} signal events of {total_sn_in_sr:d}")
    N_exp = epsB*np.sum(n_sr)*(1+Rsys)
    #sigma_stat = np.sqrt(1/N_exp+1/total_selected_cr)
    #sigma = np.sqrt(sigma_sys**2+sigma_stat**2)
    #S = (total_selected_sr-N_exp)/(N_exp*sigma)
    
    sigma_exp_stat = 1/np.sqrt(total_selected_cr)
    sigma_exp = N_exp*np.sqrt(sigma_exp_stat**2+sigma_sys**2)
    sigma_stat = np.sqrt(1/N_exp+1/total_selected_cr)
    #Asimov Estimate
    #S = np.sqrt(2*(total_selected_sr*np.log(total_selected_sr*(N_exp+sigma_exp**2)/(N_exp**2+total_selected_sr*sigma_exp**2))
    #               -N_exp/sigma_exp**2*np.log1p(sigma_exp**2*(total_selected_sr-N_exp)/(N_exp*(N_exp+sigma_exp**2)))))
    S = np.sqrt(max(0,2*(total_selected_sr*np.log(total_selected_sr*(N_exp+sigma_exp**2)/(N_exp**2+total_selected_sr*sigma_exp**2))
                   -N_exp/sigma_exp**2*np.log1p(sigma_exp**2*(total_selected_sr-N_exp)/(N_exp*(N_exp+sigma_exp**2))))))
    sigma_simple = np.sqrt(N_exp+sigma_exp**2)
    S_simple = (total_selected_sr-N_exp)/sigma_simple
    sigma = np.sqrt(total_selected_cr*(1+np.sum(n_sr)/np.sum(n_cr)))
    alpha = (total_selected_sr-N_exp)/sigma
    result.update({"eps_b": epsB,
            "cr": {"background": total_bg_in_cr, "signal": total_sn_in_cr,
                    "selected": {"total": total_selected_cr, "background": np.sum(n_selected_bg_in_cr), "signal": np.sum(n_selected_sn_in_cr)}},
            "sr": {"background": total_bg_in_sr, "signal": total_sn_in_sr,
                    "selected": {"total": total_selected_sr, "background": np.sum(n_selected_bg_in_sr), "signal": np.sum(n_selected_sn_in_sr), "expected": N_exp}},
            "sigma": {"stat": sigma_stat, "sys": sigma_sys, "exp_stat": sigma_exp_stat, "exp": sigma_exp, "simple": sigma_simple, "old": sigma},
            "S": S, "S_simple": S_simple,"significance": alpha,
            "Rsys": Rsys})
    if not quiet:
        print(f"\t\tExpected events in SR: {N_exp}")
        print(f"\tsigma_sys: {sigma_sys:.3f}, sigma_stat: {sigma_stat:.3f}")
        print(f"\tsigma_exp_stat: {sigma_exp_stat:.3f}, sigma_exp: {sigma_exp:.3f}")
        print(f"\tS-measure: {S:.3f}")
        print(f"\tsigma_old: {sigma:.3f}")
        print(f"\tsignificance: {alpha:.3f}")
    return result

'''
def evaluate_sr_vs_cr_new(y_pred_cr, y_true_cr, y_pred_sr, y_true_sr, epsB=1e-3, Rsys=0, sigma_sys=0, quiet=False):
    to_select = int(np.round(y_pred_cr.shape[0]*epsB))
    thres = np.partition(y_pred_cr, -to_select)[-to_select]
    is_selected_cr = y_pred_cr>=thres
    n_selected_cr = np.sum(is_selected_cr)
    is_background_cr = y_true_cr==0
    n_background_in_cr = np.sum(is_selected_cr&is_background_cr)
    n_signal_in_cr = np.sum(is_selected_cr& ~is_background_cr)
    if not quiet:
        print(f"At eps_b={epsB:f}%, selected:")
        print("\tIn CR:")
        print(f"\t\t{n_background_in_cr:d} background events of {np.sum(is_background_cr):d}")
        print(f"\t\t{n_signal_in_cr:d} signal events of {np.sum(~is_background_cr):d}")
    
    is_selected_sr = y_pred_sr>=thres
    n_selected_sr = np.sum(is_selected_sr)
    is_background_sr = y_true_sr==0
    n_background_in_sr = np.sum(is_selected_sr&is_background_sr)
    n_signal_in_sr = np.sum(is_selected_sr& ~is_background_sr)
    if not quiet:
        print("\tIn SR:")
        print(f"\t\t{np.sum(is_selected_sr):d} events selected in total")
        print(f"\t\t{n_background_in_sr:d} background events of {np.sum(is_background_sr):d}")
        print(f"\t\t{n_signal_in_sr:d} signal events of {np.sum(~is_background_sr):d}")
    #calculate N_exp
    N_exp = epsB*y_pred_sr.shape[0]*(1+Rsys)
    sigma_exp_stat = 1/np.sqrt(n_selected_cr)
    sigma_exp = N_exp*np.sqrt(sigma_exp_stat**2+sigma_sys**2)
    sigma_stat = np.sqrt(1/N_exp+1/n_selected_cr)
    #Asimov Estimate
    S = np.sqrt(2*(n_selected_sr*np.log(n_selected_sr*(N_exp+sigma_exp**2)/(N_exp**2+n_selected_sr*sigma_exp**2))
                   -N_exp/sigma_exp**2*np.log1p(sigma_exp**2*(n_selected_sr-N_exp)/(N_exp*(N_exp+sigma_exp**2)))))
    if not quiet:
        print(f"\t\tExpected events in SR: {N_exp}")
        print(f"\t\tExpected events in SR: {N_exp}")
        print(f"\tsigma_sys: {sigma_sys:.3f}, sigma_stat: {sigma_stat:.3f}")
        print(f"\tsigma_exp_stat: {sigma_exp_stat:.3f}, sigma_exp: {sigma_exp:.3f}")
        print(f"\tS-measure: {S:.3f}")
    return {"eps_b": epsB,
            "cr": {"background": np.sum(is_background_cr), "signal": np.sum(~is_background_cr),
                    "selected": {"total": np.sum(is_selected_cr), "background": n_background_in_cr, "signal": n_signal_in_cr}},
            "sr": {"background": np.sum(is_background_sr), "signal": np.sum(~is_background_sr),
                    "selected": {"total": np.sum(is_selected_sr), "background": n_background_in_sr, "signal": n_signal_in_sr}},
            "sigma": {"stat": sigma_stat, "sys": sigma_sys, "exp_stat": sigma_exp_stat, "exp": sigma_exp},
            "S": S}
'''