# ROC code
# A Sketch

# set up the polygraph

import numpy as np

def frange(start, stop, step):
    i = start
    while i <= stop:
        yield i
        i += step
        
def dbz(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0
    
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


data = roc_table

allData = []

for tauIn in frange(0, 1.01, .05):

    # add threshold(s) to calculate "prediction"
    tau = tauIn
    newScore = pd.DataFrame(data['Predicted'] > tau) + 0
    data2 = pd.concat([data, newScore],axis=1,ignore_index=True)
    data2.columns = ['Probability', 'Actual', 'Predicted']

    Event=data2.loc[(data2['Predicted']==True)].shape[0]
    nonEvent=data2.loc[(data2['Predicted']==False)].shape[0]

    correctEvent = data2.loc[(data2['Actual']==True) & (data2['Predicted']==True)].shape[0]
    correctNonEvent = data2.loc[(data2['Actual']==False) & (data2['Predicted']==False)].shape[0]
    incorrectEvent = data2.loc[(data2['Actual']==False) & (data2['Predicted']==True)].shape[0]
    incorrectNonEvent = data2.loc[(data2['Actual']==True) & (data2['Predicted']==False)].shape[0]

    n = correctEvent + correctNonEvent + incorrectEvent + incorrectNonEvent

    # % Correct aka Accuracy (big mistake)
    correctPerc = (correctEvent + correctNonEvent)/n

    round(correctPerc, 3)

    # Track number of Pos/Neg Events

    numPos = data2.loc[(data2['Actual']==1)].shape[0]
    numNeg = data2.loc[(data2['Actual']==0)].shape[0]

    # sens
    tpr = round(correctEvent/numPos, 3)

    # spec
    tnr = round(correctNonEvent/numNeg, 3)

    # fpr
    fpr = round(dbz(incorrectEvent, (correctEvent + incorrectEvent)), 3)

    # fnr
    fnr = round(dbz(incorrectNonEvent, (correctNonEvent + incorrectNonEvent)), 3)

    outDF = []
    outDF = [tau, n, numPos, numNeg, Event, nonEvent, correctEvent, correctNonEvent, incorrectEvent, incorrectNonEvent, correctPerc, tpr, tnr, fpr, fnr, 1-tnr]
    outDF = pd.DataFrame(data = outDF)
    outDFTrans = outDF.T
    outDFTrans.columns = ['tau', 'n', 'numPos', 'numNeg', 'Event','nonEvent', 'correctEvent', 'correctNonEvent', 'incorrectEvent', 'incorrectNonEvent', 'Acc', 'Sens', 'Spec', 'fpr', 'fnr', '1-Spec']
    
    allData.append(outDFTrans)
    
finalData = pd.concat(allData)

finalData

finalData.plot(kind='scatter', x='1-Spec', y='Sens',color='red')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Specificity')
plt.xlabel('1 - Sensitivity')
# add diagonal #
x = np.linspace(0,1,100)
y = x
plt.plot(x, y, ':b', label = 'Reference')
plt.grid()
plt.title('ROC Curve')
plt.show()
