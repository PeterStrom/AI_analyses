from sklearn.metrics import roc_auc_score
import pickle
from argparse import Namespace
import numpy as np
import pandas as pd

np.random.seed(1)

# Load data
with open('../WSI2/final/results_testdata.pkl', "rb") as fp:  # Unpickling the results
    results = pickle.load(fp)
S_t = Namespace(**results)

S_t.Y_slide['Y_valid'].loc[S_t.Y_slide['Y_valid'].slide == '13103120 3C',
                         ['ISUP', 'cx', 'any4']] = [1, 1, 1]
S_t.Y_man.loc['13103120', ['ISUP', 'cx', 'any4']] = [1, 1, 1]

with open('../WSI2/final/results_sens99_p0.05.pkl', "rb") as fp:   # Unpickling the results
    results = pickle.load(fp)
S_e = Namespace(**results)


# Point estimate AUC for test and external.
pr_core_t = 1 - S_t.pr_slide_isup[:, 0]
pr_core_e = 1 - S_e.pr_slide_isup[:, 0]

auc_c_t = roc_auc_score(S_t.Y_slide['Y_valid'].cx, pr_core_t)
auc_c_e = roc_auc_score(S_e.Y_slide['Y_valid'].cx, pr_core_e)

auc_m_t = roc_auc_score(S_t.Y_man.cx, S_t.pr_man_isup.cx)
auc_m_e = roc_auc_score(S_e.Y_man.cx, S_e.pr_man_isup.cx)


# Confidence intervals bootstrap
def bootstrap_auc(y_true, y_pred, nsamples=1000):
    n = len(y_true)
    num = [np.random.randint(n, size=n) for _ in range(nsamples)]
    auc = [roc_auc_score(y_true[i], y_pred[i]) for i in num]
    return np.percentile(auc, (2.5, 97.5))


auc_ci_c_t = bootstrap_auc(S_t.Y_slide['Y_valid'].cx, pr_core_t, 1000)
auc_ci_c_e = bootstrap_auc(S_e.Y_slide['Y_valid'].cx, pr_core_e, 1000)

auc_ci_m_t = bootstrap_auc(S_t.Y_man.cx, S_t.pr_man_isup.cx, 1000)
auc_ci_m_e = bootstrap_auc(S_e.Y_man.cx, S_e.pr_man_isup.cx, 1000)

# Round
auc_ci_c_t = ['{:.3f}'.format(x) for x in auc_ci_c_t]
auc_ci_c_e = ['{:.3f}'.format(x) for x in auc_ci_c_e]

auc_ci_m_t = ['{:.3f}'.format(x) for x in auc_ci_m_t]
auc_ci_m_e = ['{:.3f}'.format(x) for x in auc_ci_m_e]


# Point estimate correlation and confidence intervals bootstrap
def bootstrap_mm(y_true, y_pred, nsamples=1000):
    n = len(y_true)
    num = [np.random.randint(n, size=n) for _ in range(nsamples)]
    auc = [np.corrcoef(y_true[i], y_pred[i])[1, 0] for i in num]
    return np.percentile(auc, (2.5, 97.5))


iscx_c_t = np.where(S_t.Y_slide['Y_valid'].cx.isin([0]), False, True)
iscx_c_e = np.where(S_e.Y_slide['Y_valid'].cx.isin([0]), False, True)
iscx_m_t = np.where(S_t.Y_man.cx.isin([0]), False, True)
iscx_m_e = np.where(S_e.Y_man.cx.isin([0]), False, True)

# reshape arrays to pandas
y_pred_mm_c_p_t = pd.Series(S_t.pr_slide_mmca)[iscx_c_t].reset_index(drop=True)
y_true_mm_c_p_t = pd.Series(S_t.Y_slide['Y_valid'].CA_length)[iscx_c_t].reset_index(drop=True)

y_pred_mm_c_p_e = pd.Series(S_e.pr_slide_mmca)[iscx_c_e].reset_index(drop=True)
y_true_mm_c_p_e = pd.Series(S_e.Y_slide['Y_valid'].CA_length)[iscx_c_e].reset_index(drop=True)

# calculate
mm_ci_c_a_t = bootstrap_mm(S_t.pr_slide_mmca, S_t.Y_slide['Y_valid'].CA_length, 1000)
mm_ci_c_p_t = bootstrap_mm(y_pred_mm_c_p_t, y_true_mm_c_p_t, 1000)
mm_ci_c_a_e = bootstrap_mm(S_e.pr_slide_mmca, S_e.Y_slide['Y_valid'].CA_length, 1000)
mm_ci_c_p_e = bootstrap_mm(y_pred_mm_c_p_e, y_true_mm_c_p_e, 1000)

mm_ci_m_a_t = bootstrap_mm(S_t.pr_man_mmca, S_t.Y_man.CA_length, 1000)
mm_ci_m_p_t = bootstrap_mm(S_t.pr_man_mmca[iscx_m_t], S_t.Y_man.CA_length[iscx_m_t], 1000)
mm_ci_m_a_e = bootstrap_mm(S_e.pr_man_mmca, S_e.Y_man.CA_length, 1000)
mm_ci_m_p_e = bootstrap_mm(S_e.pr_man_mmca[iscx_m_e], S_e.Y_man.CA_length[iscx_m_e], 1000)

# Round
mm_ci_c_a_t = ['{:.2f}'.format(x) for x in mm_ci_c_a_t]
mm_ci_c_p_t = ['{:.2f}'.format(x) for x in mm_ci_c_p_t]
mm_ci_c_a_e = ['{:.2f}'.format(x) for x in mm_ci_c_a_e]
mm_ci_c_p_e = ['{:.2f}'.format(x) for x in mm_ci_c_p_e]

mm_ci_m_a_t = ['{:.2f}'.format(x) for x in mm_ci_m_a_t]
mm_ci_m_p_t = ['{:.2f}'.format(x) for x in mm_ci_m_p_t]
mm_ci_m_a_e = ['{:.2f}'.format(x) for x in mm_ci_m_a_e]
mm_ci_m_p_e = ['{:.2f}'.format(x) for x in mm_ci_m_p_e]


# Print results:
print('AUC 95CI core test: ', auc_ci_c_t)
print('AUC 95CI core exte: ', auc_ci_c_e)
print('AUC 95CI subj test: ', auc_ci_m_t)
print('AUC 95CI subj exte: ', auc_ci_m_e)

print('mm_corr 95CI core test: ', mm_ci_c_a_t)
print('mm_corr 95CI core test (only positive): ', mm_ci_c_p_t)
print('mm_corr 95CI core exte: ', mm_ci_c_a_e)
print('mm_corr 95CI core exte (only positive): ', mm_ci_c_p_e)

print('mm_corr 95CI subj test: ', mm_ci_m_a_t)
print('mm_corr 95CI subj test (only positive): ', mm_ci_m_p_t)
print('mm_corr 95CI subj exte: ', mm_ci_m_a_e)
print('mm_corr 95CI subj exte (only positive): ', mm_ci_m_p_e)
