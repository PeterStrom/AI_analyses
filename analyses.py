import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import Namespace
import evaluationslide as es
from sklearn.metrics import confusion_matrix

with open('../WSI2/final/results_testdata.pkl', "rb") as fp:  # Unpickling the results
    results = pickle.load(fp)

S = Namespace(**results)

# =============================================================================
# Table 2
# =============================================================================
t2 = es.table_sens_wrap(S, [0.998, 0.995, 0.9925, 0.99])
t2.to_csv('../WSI2/table2.csv')

# =============================================================================
# ROCAUC
# =============================================================================
es.plot_roc(S,
            figsize=(5, 5),
            xlim=[100, 0],
            ylim=[0, 100],
            dec=3,
            save_format='.png',
            dpi=120)

# =============================================================================
# Confusion matrix
# =============================================================================
cnf_matrix = confusion_matrix(S.Y_slide['Y_valid'].ISUP, S.cl_slide_isup)

plt.figure()
es.plot_confusion_matrix(cnf_matrix,
                         title='Confusion matrix',
                         cmap=plt.cm.Blues)
plt.show()
plt.savefig('confusion_test.pdf', bbox_inches='tight', dpi=1200)

# =============================================================================
# Plot mm
# if ci != None, then ci_manuscript.py must be executed first.
# =============================================================================
es.plot_mm(S,
           core_level=True,
           figsize=(7, 7),
           save_format='.svg',
           dpi=180,
           ci=[mm_ci_c_a_t, mm_ci_c_p_t])

es.plot_mm(S,
           core_level=False,
           figsize=(7, 7),
           save_format='.svg',
           dpi=180,
           ci=[mm_ci_m_a_t, mm_ci_m_p_t])

# =============================================================================
# Table Kappa Test data
# =============================================================================
t_kappa = es.kappa_test(S, dec=2)
print(t_kappa)

# =============================================================================
# Imagebase
# =============================================================================
# Add AI to Imagebase
df = pd.read_csv('../IMAGEBASE/FinalResultsProstate.csv')
df_AI = pd.DataFrame({'Path no': S.Y_slide['Y_valid'].slide, 'AI': S.cl_slide_isup})
df_AI['Path no'] = df_AI['Path no'].str.lower()
newstring = [x[:2] + x[3:] for x in df_AI['Path no']]  # Remove the 8.
df_AI['Path no'] = newstring
df_AI = df_AI[df_AI['Path no'] != '1400110 1a']  # Wrong, not the same as in ImageBase.
df = pd.merge(df, df_AI, how='inner', on='Path no')

names = ['Evans', 'Delahunt', 'Pan', 'CMG', 'Berney', 'Grignon', 'Comperat',
         'Kristiansen', 'Samaratunga', 'Kench', 'McKenney', 'Srigley', 'Oxley',
         'Leite', 'Iczkowski', 'Egevad', 'Zhou', 'Varma', 'Humphrey', 'Fine',
         'Hiroyuki', 'Kwast', 'Tsuzuki', 'AI']
df = df.loc[:, names]
df_imagebase = df.copy()

# =============================================================================
# Figure 'Bubble chart'.
# =============================================================================
df['Case'] = df.index + 1
df = pd.melt(df, id_vars=['Case'])
df.columns = ['Case', 'Voter', 'ISUP']

color = df[(df.Voter == 'AI')].ISUP
color = color.reset_index(drop=True)

df = df.groupby(['Case', 'ISUP'], as_index=False).agg('count')
df['c'] = np.repeat('b', len(df))

for i, j in enumerate(color):
    a = (df.Case == i + 1) & (df.ISUP == j)
    df['c'].where(~a, 'r', inplace=True)

df['sum'] = df.ISUP * df.Voter
order = df.groupby('Case')['sum'].sum()
order = list(order)

df_o = pd.DataFrame({'Case': range(1, 88), 'Order': order})

df_o.columns = ['Case', 'ISUP_mean']

df = pd.merge(df, df_o, how='left', on='Case')
df = df.sort_values('ISUP_mean', ascending=False)
df['Case'] = df['Case'].astype(str)

fig, ax = plt.subplots(figsize=(4, 40))
plt.scatter(y=df.Case,
            x=df.ISUP,
            s=df.Voter * 23,
            alpha=0.5,
            color='b')
df_only_ai = df[df.c == 'r']
plt.scatter(y=df_only_ai.Case,
            x=df_only_ai.ISUP,
            s=23,
            color='r')
plt.ylabel("Case ID")
plt.xlabel("ISUP Score")
plt.savefig('../Bubbles_ex.pdf', bbox_inches='tight', dpi=120)

# =============================================================================
# Figure 'Kappa'.
# =============================================================================
df = df_imagebase
k_or = es.imagebase_kappas(df, color_ai='0', color_patholgists='r', group=False)
k_gr = es.imagebase_kappas(df, color_ai='0', color_patholgists='b', group=True)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)

rank = list(reversed(range(1, 25)))
plt.scatter(rank,
            k_or["Kappa"],
            c=k_or['Color'],
            alpha=0.5,
            label="ISUP 1 to 5")
plt.gca().invert_xaxis()
plt.scatter(rank,
            k_gr["Kappa"],
            c=k_gr['Color'],
            alpha=0.5,
            label="ISUP 1, 2-3, 4-5")

plt.xlabel("Pathologist ranking")
plt.ylabel("Cohen\'s Kappa")
plt.yticks(np.arange(0.5, .8, step=0.05))
plt.xticks(np.arange(1, 26, 6))
ax.legend(loc='lower right')
plt.savefig('../Kappas.png', bbox_inches='tight', dpi=120)
