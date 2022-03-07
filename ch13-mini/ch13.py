import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from dtreeviz.trees import *
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF, renderPM
from sklearn.ensemble import RandomForestClassifier
import time
import numpy as np
import matplotlib.pyplot as plt
from pdpbox import pdp, get_dataset, info_plots


def to_bool(x):
	if x == 'Good': return 1
	elif x == 'Bad': return 0

GermanCredit = pd.read_csv("GermanCredit.csv.zip")
x = GermanCredit[GermanCredit.columns.drop('Class')]
y = GermanCredit['Class']
y = y.apply(to_bool)
# print(y2)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x.values, y.values,train_size = 0.75,random_state=5)

params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}
dt = DecisionTreeClassifier()
cm2 = GridSearchCV(estimator=dt, param_grid=params, cv=4, n_jobs=-1, verbose = 1,scoring = "accuracy")

cm2.fit(Xtrain, Ytrain)
dt_best = cm2.best_estimator_
YPred = dt_best.predict(Xtest)
accuracy = accuracy_score(Ytest, YPred)
# YPred = cm2.predict(Xtest) 
# accuracy = accuracy_score(Ytest,YPred)
print(accuracy)

# dt_best = cm2.best_estimator_
# dt_best.predict(Xtest)

viz = dtreeviz(dt_best, Xtrain, Ytrain, target_name='Class', feature_names=list(x.columns))
viz.save("decision_tree.svg")
drawing = svg2rlg("decision_tree.svg")
renderPDF.drawToFile(drawing, "decision_tree.pdf")

forest = RandomForestClassifier(random_state = 0) 
forest.fit(Xtrain, Ytrain)
YPred2 = forest.predict(Xtest) 
accuracy2 = accuracy_score(Ytest,YPred2)
print(accuracy2)
start_time = time.time()
importances = forest.feature_importances_
test_list = [tree.feature_importances_ for tree in forest.estimators_]
std = np.std(test_list, axis=0)
# std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0, dtype = np.float64)
elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
forest_importances = pd.Series(importances, index=list(x.columns))

fig, ax = plt.subplots()
# print(std)
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.savefig("forrest.pdf")
# plt.show()

# amount and duration
# print(type(Xtrain))
pdp_amount = pdp.pdp_isolate(model=forest, dataset=x, model_features=list(x.columns), feature='Amount')
fig2, ax2 = pdp.pdp_plot(pdp_amount, 'Amount', plot_lines=True, frac_to_plot=100)
# ax2.set_title("pdp amount")
fig2.savefig("pdp_amount.pdf")

pdp_duration = pdp.pdp_isolate(model=forest, dataset=x, model_features=list(x.columns), feature='Duration')
fig3, ax3 = pdp.pdp_plot(pdp_duration, 'Duration', plot_lines=True, frac_to_plot=100)
# ax3.set_title("pdp duration")
fig3.savefig("pdp_duration.pdf")
