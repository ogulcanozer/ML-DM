#_______________________________________________________________________________
# CE802 Machine Learning and Data Mining | Ogulcan Ozer. | 25 December 2018
#_______________________________________________________________________________
import graphviz, os, sklearn, numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score , recall_score , precision_score , classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

kf_CV = KFold(n_splits=5, shuffle=True)
s_p = 2/3
#-------------------------------------------------------------------------------
# Functions
#-------------------------------------------------------------------------------
def read_csv():
    dft = pd.read_csv(os.path.join(os.path.dirname(__file__),"data_ass.csv"))
    dft = shuffle(dft)

    data_scale = dft.iloc[:,:-1]
    scaler = StandardScaler()
    scaler.fit(data_scale)

    train= dft.iloc[:(int(len(dft.index)*s_p)),:-1]
    test= dft.iloc[(int(len(dft.index)*s_p)):,:-1]
    tr_target = dft.iloc[:int(len(dft.index)*s_p),-1].astype(int)
    te_target = dft.iloc[int(len(dft.index)*s_p):,-1].astype(int)
    dfp = pd.read_csv(os.path.join(os.path.dirname(__file__),"data_predict.csv"))
    predict = dfp.iloc[:,:-1]

    train = pd.DataFrame(scaler.transform(train))
    test = pd.DataFrame(scaler.transform(test))
    predict = pd.DataFrame(scaler.transform(predict))
    return train, test, tr_target, te_target, predict

def _svm(tr, ta, pr):
    #Set the local variables
    data_train, data_target, data_predict = tr, ta, pr
    #Set the parameter ranges for the gridsearch
    Cs = np.logspace(-1, 3, 9)
    Gs = np.logspace(-7, -0, 8)
    #Create an svm classifier with placeholder parameters.
    cls_svm = SVC(gamma=0.0001, C=100.)
    #Initialize gridsearch with our parameters
    cls_svm = GridSearchCV(estimator=cls_svm, param_grid=dict(C=Cs, gamma=Gs),return_train_score=True,scoring='accuracy',cv=kf_CV, n_jobs=-1)
    #Fit the data
    cls_svm.fit(data_train, data_target)
    cls_tostring(cls_svm)
    return cls_svm


def _pdt(tr, ta, pr):
    #Set the local variables
    data_train, data_target, data_predict = tr, ta, pr
    #Set the parameter ranges for the gridsearch
    md = np.arange(1, 20)
     #Create a decision tree with placeholder parameters.
    cls_dt = tree.DecisionTreeClassifier(criterion = "entropy", max_depth=3)
    #Initialize gridsearch with our parameters
    cls_dt = GridSearchCV(estimator=cls_dt, param_grid=dict(max_depth=md),return_train_score=True,scoring='accuracy',cv=kf_CV, n_jobs=-1)
    #Fit the data
    cls_dt.fit(data_train, data_target)
    cls_tostring(cls_dt)
    return cls_dt

def _mlp(tr, ta, pr):
    #Set the local variables
    data_train, data_target, data_predict = tr, ta, pr
    #Set the parameter ranges for the gridsearch
    a = 10.0 ** -np.arange(1, 7)
    h = [(2,),(4,),(14,),(28,),(42,),(56,)]
    s = ['sgd','adam']
    #Create an mlp classifier using stochastic gradient descent with momentum
    cls_mlp = MLPClassifier(solver='sgd', learning_rate= 'constant',momentum= .9, nesterovs_momentum= False, learning_rate_init= 0.2, alpha=1e-5, hidden_layer_sizes=(28,), random_state=1)
    #Initialize gridsearch with our parameters
    cls_mlp = GridSearchCV(estimator=cls_mlp, param_grid=dict(alpha=a, hidden_layer_sizes=h, solver=s),return_train_score=True,scoring='accuracy',cv=kf_CV, n_jobs=-1)
    #Fit the data
    cls_mlp.fit(data_train, data_target)
    cls_tostring(cls_mlp)
    return cls_mlp

def cls_tostring(cls):
        best_s = cls.best_score_
        best_p = cls.best_params_
        print("Best training score for the gridsearch of the "+cls.best_estimator_.__class__.__qualname__+":")
        print(np.amax(cls.cv_results_['mean_train_score']))
        print(cls.cv_results_)
        print("Best validation score for the gridsearch of the "+cls.best_estimator_.__class__.__qualname__+":")
        print(best_s)
        print("Parameters for the best score: ")
        print(best_p)
        return

def _compare(clss, test, target):
        results = []
        for i in range(0,len(clss)):
            results.append(pd.DataFrame(clss[i].predict(test)))

        for i in range(0,len(results)):
            print("Test results of the "+clss[i].best_estimator_.__class__.__qualname__+":")
            cls_report = classification_report(data_te_target,results[i])
            print(cls_report)
            print("accuracy : " + str(accuracy_score(data_te_target,results[i])))
            tn, fp, fn, tp = confusion_matrix(data_te_target, results[i]).ravel()
            print(pd.DataFrame(confusion_matrix(data_te_target, results[i], labels=[0, 1]), index=['a = T', 'a = F'], columns=['p = T', 'p = F']))

        print("Differences between algorithms :")
        done = set()
        for i in range(0,len(clss)):
            for j in range(0,len(clss)):
                if(i != j and (i+j not in done)):
                    print(clss[i].best_estimator_.__class__.__qualname__+" and "+clss[j].best_estimator_.__class__.__qualname__)
                    mn = mc_nemar(results[i],results[j],target)
                    done.add(i+j)
                    print('statistic=%.8f, p-value=%.8f' % (mn.statistic, mn.pvalue))
                    alpha = 0.05
                    if mn.pvalue > alpha:
                    	print('Difference is non-significant')
                    else:
                    	print('Significant difference')

        roc_auc(clss,results,target)

def mc_nemar(c1,c2,target):

    ss = sf = fs = ff = 0
    for i in range (0,len(target)):
        t = int(target.iloc[i])
        c1r = int(c1.iloc[i])
        c2r = int(c2.iloc[i])
        if((t == c1r) and (t == c2r)):
            ss = ss + 1
        elif((t != c1r) and (t != c2r)):
            ff = ff + 1
        elif((t==c1r) and (t != c2r)):
            sf = sf + 1
        else:
            fs = fs + 1
    table = [[ss, sf],[fs, ff]]
    result = mcnemar(table, exact=False, correction=True)
    return result

def roc_auc(clss, res, target):
    fpr = dict()
    tpr = dict()
    auc_roc = dict()
    for i in range(0,3):
        fpr[i], tpr[i], _ = roc_curve(target, res[i])
        auc_roc[i] = auc(fpr[i],tpr[i])

    plt.figure()
    for i in range(0,len(res)):
        lbl = f"ROC curve of {clss[i].best_estimator_.__class__.__qualname__} (area= {auc_roc[i]})"
        plt.plot(fpr[i], tpr[i], lw=2, label = lbl )
        print('AUC of '+clss[i].best_estimator_.__class__.__qualname__)
        print(auc_roc[i])

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves of each classifier')
    plt.legend(loc="lower right")
    plt.show()

#-------------------------------------------------------------------------------
# Main program.
#-------------------------------------------------------------------------------

#Read and parse the training and prediction data.
data_train, data_test, data_tr_target,data_te_target, data_predict = read_csv()

#Classifier list to hold best fitted classifiers.
classifiers = []

#Append the returned classifiers.
classifiers.append(_svm(data_train, data_tr_target, data_predict))
classifiers.append(_pdt(data_train, data_tr_target, data_predict))
classifiers.append(_mlp(data_train, data_tr_target, data_predict))

#Compare the classifiers.
_compare(classifiers,data_test,data_te_target)

results = pd.read_csv(os.path.join(os.path.dirname(__file__),"data_predict.csv"))
result_predict = pd.DataFrame(classifiers[0].best_estimator_.predict(data_predict))
for i in range(0,len(result_predict.index)):
    if(result_predict.iloc[i,0]==1):
        results.loc[i,'Class']= 'True'
    else:
        results.loc[i,'Class']= 'False'

results.to_csv('results.csv',sep=',',index=False)

#-------------------------------------------------------------------------------
# End of Program
#-------------------------------------------------------------------------------
