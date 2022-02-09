import numpy as np
import pandas as pd

def compute_priors(y):
    value_counts = y.value_counts().sort_index()
    total_values = len(y)
    indexes = [y.name+"="+str(v) for v in value_counts.index.tolist()]
    priors = dict(zip(indexes, value_counts / total_values))
    return priors

def specific_class_conditional(x,xv,y,yv):
    
    likelihoods = {}
    priors = {}
    
    for ux in x.unique():
    
        binned, bins = pd.cut(y.loc[x==ux], 5, retbins=True)
        binned_counts = binned.value_counts()
        bin_value = pd.cut([yv], bins)[0]
        if bin_value in binned_counts.index:
            count = binned_counts.loc[bin_value]
        else:
            count = 0
        likelihoods[ux] = count / sum(binned_counts)
        priors[ux] = sum(binned_counts) / len(x)
    
    denom = 0
    for k in priors.keys():
        denom += (priors[k]*likelihoods[k])
    
    classcond = likelihoods[xv]*priors[xv] / denom
    return classcond


def class_conditional(X,y):
    probs = {}
    for eachy in y.unique():
        for col in X.columns:
            for eachx in X[col].unique():
                probs[col + "=" + str(eachx) + "|" + y.name + "=" + str(eachy)] = specific_class_conditional(X[col], eachx, y, eachy)
    return probs

def posteriors(probs,priors,x):
    post_probs = {}
    denom = 0
    for k in priors.keys():
        numerator = 1
        postkey = ""
        for idx in x.index:
            postkey += idx + "=" + str(x[idx]) + ","
            probkey = idx + "=" + str(x[idx]) + "|" + k
            if probkey not in probs:
                numerator *= 0
            else:
                numerator *= probs[probkey]
        numerator*=priors[k]
        post_probs[k + "|" + postkey[:-1]] = numerator
        denom+=numerator
    for k, v in post_probs.items():
        if denom != 0:
            post_probs[k] = v / denom
        else:
            post_probs[k] = 1 / len(list(priors.keys()))
    return post_probs

def train_test_split(X,y,test_frac=0.5):
    inxs = list(range(len(y)))
    np.random.shuffle(inxs)
    X = X.iloc[inxs,:]
    y = y.iloc[inxs]
    xsplit = round(len(X)*test_frac)
    ysplit = round(len(y)*test_frac)
    Xtrain,ytrain,Xtest,ytest = X.iloc[:xsplit, :], y.iloc[:ysplit], X.iloc[xsplit:, :], y.iloc[ysplit:]
    return Xtrain,ytrain,Xtest,ytest

def exercise_6(Xtrain,ytrain,Xtest,ytest):
    probs = class_conditional(Xtrain, ytrain)
    priors = compute_priors(ytrain)
    ypred = []
    for idx in range(len(Xtest)):
        posterior = posteriors(probs, priors, Xtest.iloc[idx])
        k = max(posterior, key=posterior.get)
        pred = round(float((k.split("|")[0]).split("=")[-1]))
        ypred.append(pred)
    ypred = np.array(ypred)
    correct = np.sum(ypred == ytest.to_numpy())
    accuracy = correct / len(ytest)
    return accuracy

def test_based_feature_importance(Xtrain,ytrain,Xtest,ytest, npermutations = 10):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    print(orig_accuracy)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
            new_accuracy = exercise_6(Xtrain, ytrain, Xtest2, ytest)
            print(new_accuracy)
            feature_importance = orig_accuracy - new_accuracy
            print(feature_importance)
            importances[col] += feature_importance
            
        importances[col] = importances[col]/npermutations
    return importances

def train_based_feature_importance(Xtrain,ytrain,Xtest,ytest, npermutations = 20):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtrain2 = Xtrain.copy()
            Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values
            new_accuracy = exercise_6(Xtrain2, ytrain, Xtest, ytest)
            feature_importance = orig_accuracy - new_accuracy
            importances[col] += feature_importance
            
        importances[col] = importances[col]/npermutations
    return importances
