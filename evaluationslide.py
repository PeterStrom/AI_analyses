import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn import metrics


def table_sens(sens, pred, y, fp_fn=False):
    """Generate values for each row in table about sensitivity and specificity.
    Args:
        sens: (num) choose a fix sensitivity.
        pred: (array) predictions where first column correspond to Benign.
        y: (DataFrame) from agg_features().
        fp_fn: (bool) if True, only return FP and FN slides (mainly for
               visualisation).
    Return: sensitivity, specificity and list of avioded bipsies and percent
        for each ISUP score (including Benign).
    """
    if pred.ndim == 1:
        pred_cx = pred
    else:
        pred_cx = 1 - pred[:, 0]
    fpr, tpr, tr = metrics.roc_curve(y.cx, pred_cx)

    index = np.argmax(tpr > sens)
    sens = tpr[index]
    roc_y = sens
    roc_x = fpr[index]
    thres = tr[index]
    avoid = y.ISUP[pred_cx < thres]
    review = y.ISUP[pred_cx >= thres]
    n_tp = np.sum(review > 0)

    if fp_fn:
        fp = y.loc[(pred_cx >= thres) & (y.cx == 0), ['slide', 'ISUP']]
        fn = y.loc[(pred_cx < thres) & (y.cx == 1), ['slide', 'ISUP']]
        return fp, fn

    unique_tot, counts_tot = np.unique(y.ISUP, return_counts=True)
    unique, counts = np.unique(avoid, return_counts=True)
    avoid = np.column_stack((unique, counts))

    if len(unique) < 6:
        zeros = set(np.arange(6)) - set(unique)  # levels with zeros
        zeros = np.array(list(zeros))
        zero_count = np.array([0 for x in zeros])  # the zero values
        avoid_zero = np.column_stack((zeros, zero_count))
        avoid = np.row_stack((avoid, avoid_zero))  # all levels incl zeros
        avoid = avoid[avoid[:, 0].argsort()]

    # Format output
    sens = "{0:.0f} ({1:.1f})".format(n_tp, sens * 100)
    pers = avoid[:, 1] / counts_tot * 100
    pers = [" ({0:.1f})".format(x) for x in pers]
    count_avoid = [str(x) for x in avoid[:, 1]]
    avoid = [x + y for (x, y) in zip(count_avoid, pers)]

    # Man level
    pr_man = pd.DataFrame({'pr_cx': pred_cx})

    man = pd.concat([pr_man.reset_index(drop=True),
                     y.reset_index(drop=True)],
                    axis=1)
    man = man.groupby('man').agg({'cx': 'max',
                                  'any4': 'max',
                                  'pr_cx': 'max'
                                  })

    man_tp = np.sum(man[man.pr_cx < thres].cx)
    man_p = np.sum(man.cx)
    man_tpf = man_tp / man_p * 100
    man_cx = "{0:.0f} ({1:.1f})".format(man_tp, man_tpf)

    return [avoid[0], sens] + avoid[1:] + [man_cx], (roc_x, roc_y)


def table_sens_wrap(results, thresholds):
    """Produce a dataframe with one row per specificity threshold in args.
    Args:
        results: (Namespace) probabilites and outcome
        thresholds: (list) thresholds to evaluate on
    """
    n = len(thresholds)
    row = [0] * n

    for i in range(n):
        row[i], _ = table_sens(thresholds[i],
                               pred=results.pr_slide_isup,
                               y=results.Y_slide['Y_valid'])

    t2 = np.row_stack(row)
    lab = ['Avoided Ben (spec)', 'Detected cx (sens)']
    lab += ['ISUP' + str(i) for i in np.arange(1, 6)]
    lab += ['Man missed cx (sens)']
    t2 = pd.DataFrame(t2, columns=lab)
    return t2


def plot_roc(results,
             figsize=(5, 5),
             xlim=[100, 0],
             ylim=[0, 100],
             dec=2,
             save_format='.pdf',
             dpi=120):
    """Plot ROC for Cancer discrimination.
    Args:
        results: (Namespace) probabilites and outcome
        figsize: matplotlib arg
        xlim: matplotlib arg
        ylim: matplotlib arg
        dec: decimals in AUC
        save_format: if None nothing will be saved, else string (e.g. '.png')
        dpi: resolution of saved image
        """
    pr_cx = 1 - results.pr_slide_isup[:, 0]

    # Plot details
    plt.figure(figsize=figsize)

    plt.plot(ylim, xlim, color='lightgrey', linestyle='--')

    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.ylabel("Sensitivity")
    if xlim[0] > xlim[1]:
        plt.xlabel("Specificity")
    else:
        plt.xlabel("$\\bf{100 - Specificity}$\n(False Positive Fraction)")

    plt.title('ROC: Prostate Cancer Detection')

    # Core cx
    fpr, tpr, _ = metrics.roc_curve(results.Y_slide['Y_valid'].cx, pr_cx)
    auc = metrics.auc(fpr, tpr)
    tpr = tpr * 100
    fpr = (1 - fpr) * 100
    text = 'Core: any cancer = %0.' + str(dec) + 'f'
    plt.plot(fpr, tpr, 'b', label=text % auc)

    # Man cx
    fpr, tpr, _ = metrics.roc_curve(results.Y_man.cx, results.pr_man_isup.cx)
    auc = metrics.auc(fpr, tpr)
    tpr = tpr * 100
    if xlim[0] > xlim[1]:
        fpr = (1 - fpr) * 100
    else:
        fpr = fpr * 100
    text = 'Men: any cancer = %0.' + str(dec) + 'f'
    plt.plot(fpr, tpr, 'b:', label=text % auc)

    # Legend
    plt.legend(loc='lower right',
               title="$\\bf{AUC}$",
               fontsize='small',
               frameon=False)
    plt.tight_layout()

    if save_format is not None:
        plt.savefig('../ROC' + save_format, bbox_inches='tight', dpi=dpi)


def plot_mm(results,
            core_level=True,
            figsize=(12, 7),
            dot_size=2,
            save_format='.pdf',
            dpi=80,
            ci=None):
    if core_level:
        # Corr Core
        iscx_c = np.where(results.Y_slide['Y_valid'].cx.isin([0]), False, True)
        c1 = np.corrcoef(results.pr_slide_mmca,
                         results.Y_slide['Y_valid'].CA_length)[1, 0]
        c2 = np.corrcoef(results.pr_slide_mmca[iscx_c],
                         results.Y_slide['Y_valid'].CA_length[iscx_c])[1, 0]

        ticks = np.arange(0, 31, 5)

        pr = results.pr_slide_mmca

        n = len(results.Y_slide['Y_valid'].CA_length)

        noise = np.random.uniform(-.5, 0.5, n)
        y = results.Y_slide['Y_valid'].CA_length + noise
        y = np.where(y < 0, 0, y)

        title = 'Individual cores'

        t_1 = 12
        t_2 = 0

        filename = '../mm_core'

    else:
        # Corr Man
        iscx_m = np.where(results.Y_man.cx.isin([0]), False, True)
        c1 = np.corrcoef(results.pr_man_mmca,
                         results.Y_man.CA_length)[1, 0]
        c2 = np.corrcoef(results.pr_man_mmca[iscx_m],
                         results.Y_man.CA_length[iscx_m])[1, 0]

        ticks = np.arange(0, 151, 25)

        pr = results.pr_man_mmca

        y = results.Y_man.CA_length

        title = 'Total tumour burden'

        t_1 = 60
        t_2 = 0

        filename = '../mm_men'

    if not ci:
        text = 'Correlation all: {0:.2f}\nCorrelation malignant: {1:.2f}'.format(c1, c2)
    else:
        text1 = 'Correlation all: {0:.2f} ('.format(c1) + ci[0][0] + '-' + ci[0][1] + ')\n'
        text2 = 'Correlation malignant: {0:.2f} ('.format(c2) + ci[1][0] + '-' + ci[1][1] + ')'
        text = text1 + text2

    # Plot
    plt.figure(figsize=figsize)

    plt.scatter(y, pr, s=dot_size)
    plt.xlabel('Pathologist (mm)')
    plt.ylabel('Predictions (mm)')
    plt.yticks(ticks)
    plt.xticks(ticks)
    plt.text(t_1, t_2, text, bbox=dict(facecolor='white', alpha=0.2))
    plt.title(title)

    if save_format is not None:
        plt.savefig(filename + save_format, bbox_inches='tight', dpi=dpi)


def table_mm(results, dec=3):
    # Corr Core
    iscx_c = np.where(results.Y_slide['Y_valid'].cx.isin([0]), False, True)
    c1_c = np.corrcoef(results.pr_slide_mmca,
                       results.Y_slide['Y_valid'].CA_length)[1, 0]
    c2_c = np.corrcoef(results.pr_slide_mmca[iscx_c],
                       results.Y_slide['Y_valid'].CA_length[iscx_c])[1, 0]

    # Corr Man
    iscx_m = np.where(results.Y_man.cx.isin([0]), False, True)
    c1_m = np.corrcoef(results.pr_man_mmca,
                       results.Y_man.CA_length)[1, 0]
    c2_m = np.corrcoef(results.pr_man_mmca[iscx_m],
                       results.Y_man.CA_length[iscx_m])[1, 0]
    tab_mm = pd.DataFrame({'Core': [c1_c, c2_c], 'Men': [c1_m, c2_m]}, index=["All", "Only_ca"])

    return tab_mm.round(dec)


def kappa_isup(results):
    # Cohens Kappa based on the full confusion matrix.
    K_all = metrics.cohen_kappa_score(results.Y_slide['Y_valid'].ISUP,
                                      results.cl_slide_isup,
                                      weights='linear')

    iscx = np.where(results.Y_slide['Y_valid'].ISUP.isin([0]), False, True)
    K_oca = metrics.cohen_kappa_score(results.Y_slide['Y_valid'].ISUP[iscx],
                                      results.cl_slide_isup[iscx],
                                      weights='linear')
    print('All ISUP {0:.2f}, only cancer ISUP {1:.2f}'.format(K_all, K_oca))
    return K_all, K_oca


def kappa_test(results, dec):
    y = results.Y_slide['Y_valid'].ISUP.copy()
    c = results.cl_slide_isup.copy()
    iscx = np.where(y.isin([0]), False, True)

    # Cohens Kappa based on the full confusion matrix.
    k_all_5 = metrics.cohen_kappa_score(y, c, weights='linear')
    k_oca_5 = metrics.cohen_kappa_score(y[iscx], c[iscx], weights='linear')

    # Grouping ISUP 3, 4, 5
    y[y >= 3] = 3
    c[c >= 3] = 3

    k_all_3 = metrics.cohen_kappa_score(y, c, weights='linear')
    k_oca_3 = metrics.cohen_kappa_score(y[iscx], c[iscx], weights='linear')

    # Grouping ISUP 2, 3, 4, 5
    y[y >= 2] = 2
    c[c >= 2] = 2

    k_all_2 = metrics.cohen_kappa_score(y, c, weights='linear')
    k_oca_2 = metrics.cohen_kappa_score(y[iscx], c[iscx], weights='linear')

    # Present as DataFrame
    l_all = [k_all_5, k_all_3, k_all_2]
    l_oca = [k_oca_5, k_oca_3, k_oca_2]

    s = str(dec).join(['{0:.', 'f}'])
    l_all = [s.format(i) for i in l_all]
    l_oca = [s.format(i) for i in l_oca]

    d = pd.DataFrame({'All cores': l_all, 'Only Ca cores': l_oca},
                     index=['ISUP 1-5', 'ISUP 1-3+', 'ISUP 1-2+'])

    return d


def kappa_mirror_imagebase(results, dec=2, n_bootstrap=1000):
    y = results.Y_slide['Y_valid'].ISUP.copy()
    c = results.cl_slide_isup.copy()

    l_k = []
    l_g = []  # grouped kappa 1, 2-3, 4-5.

    for i in range(n_bootstrap):
        mirror_ib = pd.DataFrame({'y': y, 'c': c})

        mirror_ib = mirror_ib.sample(frac=1)

        mirror_ib_1 = mirror_ib[mirror_ib.y == 1][:20]
        mirror_ib_2 = mirror_ib[mirror_ib.y == 2][:32]
        mirror_ib_3 = mirror_ib[mirror_ib.y == 3][:16]
        mirror_ib_4 = mirror_ib[mirror_ib.y == 4][:10]
        mirror_ib_5 = mirror_ib[mirror_ib.y == 5][:13]

        mirror_ib = pd.concat((mirror_ib_1, mirror_ib_2, mirror_ib_3, mirror_ib_4, mirror_ib_5))

        # Remap ISUP values on c and y for grouped Kappa.
        gr = mirror_ib.copy()
        gr.replace([2, 3], 2, inplace=True)
        gr.replace([4, 5], 3, inplace=True)

        # Cohens Kappa based on the full confusion matrix.
        k_ib = metrics.cohen_kappa_score(mirror_ib.y, mirror_ib.c, weights='linear')
        l_k.append(k_ib)
        k_ig = metrics.cohen_kappa_score(gr.y, gr.c, weights='linear')
        l_g.append(k_ig)

    s = str(dec).join(['{0:.', 'f}'])

    def mean_sd(kappas):
        mean = np.mean(kappas)
        std = np.std(kappas)

        lower = mean - 1.96 * std
        upper = mean + 1.96 * std

        mean_str = s.format(mean)
        lower_str = s.format(lower)
        upper_str = s.format(upper)

        string = mean_str + " (" + lower_str + "-" + upper_str + ")"
        return string

    # Original scale
    string_or = mean_sd(l_k)

    # grouped scale
    string_gr = mean_sd(l_g)

    return string_or, string_gr


def imagebase_kappas(df, color_ai='r', color_patholgists='b', group=False):
    """
    Get average pairwise Cohens kappas for each of the 23 Pathologists, either
    on all ISUP 1 to 5 or grouped according to clinical relevant groups
    1, 2-3, 4-5.
    :param df: pandas DataFrame with raw data on ISUP voting.
    :param color_ai: color to use when plotting.
    :param color_patholgists: color to use when plotting.
    :param group: (bool) False: ISUP 1-5, True: 1, 2-3, 4-5.
    :return: DataFrame with columns 'Names', 'Kappa', 'Number' and 'Color'.
    """
    names = ['Evans', 'Delahunt', 'Pan', 'CMG', 'Berney', 'Grignon', 'Comperat',
             'Kristiansen', 'Samaratunga', 'Kench', 'McKenney', 'Srigley', 'Oxley',
             'Leite', 'Iczkowski', 'Egevad', 'Zhou', 'Varma', 'Humphrey', 'Fine',
             'Hiroyuki', 'Kwast', 'Tsuzuki', 'AI']
    df = df.loc[:, names]

    if group:
        df.replace([2, 3], 2, inplace=True)
        df.replace([4, 5], 3, inplace=True)

    col_pairs = list(itertools.combinations(df.columns, 2))  # unique pairs.

    kappas = []  # A kappa for each unique pair.
    for a, b in col_pairs:
        kappa = metrics.cohen_kappa_score(df[a], df[b], weights='linear')
        kappas.append(kappa)

    avg_kappas = []  # Take the average of kappas for each name.

    for i in names:
        b = [i in pair for pair in col_pairs]
        k = list(itertools.compress(kappas, b))
        m = np.mean(k)
        avg_kappas.append(m)

    # Names as integers. Name no.6 has missing and is therefor removed in agreement
    # with ImageBase main study.
    v = np.arange(1, 25)
    v = np.where(v >= 6, v + 1, v)
    v = v.astype(str)

    k = pd.DataFrame({'Names': names,
                      'Kappa': avg_kappas,
                      'Number': v.astype(str),
                      'Color': color_patholgists})
    k.loc[k['Names'] == 'AI', 'Color'] = color_ai

    k = k.sort_values('Kappa')
    return k


def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function plots the confusion matrix.
    Modified code from scikit-learn.org.
    """
    cm_tr = np.where(cm > 70, 70 + (cm - 70) / 10, cm)
    plt.imshow(cm_tr, interpolation='nearest', cmap=cmap)
    plt.title(title)
    classes = ['Benign'] + ['ISUP ' + str(i) for i in range(1, 6)]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_tr.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Pathologist grading')
    plt.xlabel('AI grading')
    plt.tight_layout()
