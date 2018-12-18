# ROC
What is ROC?

#### The ROC curve is a fundamental tool for diagnostic test evaluation.
#### In a ROC curve the true positive rate (Sensitivity) is plotted in function of the false positive rate (100-Specificity) for different #### cut-off points of a parameter.

now we need to define something :eyes:
in this case, you consider the goal of the machine to detect spam mail so 
### TP: True means machine predict right and positive means machine say it is spam and achieve the goal now
### TN: True means machine predict right and negative means machine say it isn't spam
### FP: false means machine predict wrong and positive means machine say it is spam
### FN: false means machine predict wrong and negative means machine say it isn't spam

now let's have code
if you run it you can see roc chart :sunglasses:

```python
# import requirement
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import datasets
# load data
bcd = datasets.load_breast_cancer()
x = bcd.data
y = bcd.target
# divide data and fit model
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=42)
log = LogisticRegression()
log.fit(x_train, y_train)
y_pred = log.predict(x_test)
# calculate the y_pred_prob and draw the roc chart
y_pred_prob = log.predict_proba(x_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
```

I hope this data will be useful to you.
