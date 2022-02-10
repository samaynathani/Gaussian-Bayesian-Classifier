import Lab2_helper
import numpy as np

def get_accuracy(Xtrain,ytrain,Xtest,ytest):
    probs = Lab2_helper.class_conditional(Xtrain, ytrain)
    priors = Lab2_helper.compute_priors(ytrain)
    ypred = []
    for idx in range(len(Xtest)):
        posterior = Lab2_helper.posteriors(probs, priors, Xtest.iloc[idx])
        k = max(posterior, key=posterior.get)
        pred = float((''.join(k.split("|")[0])).split("=")[-1])
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
    orig_accuracy = get_accuracy(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
            new_accuracy = get_accuracy(Xtrain, ytrain, Xtest2, ytest)
            feature_importance = orig_accuracy - new_accuracy
            importances[col] += feature_importance
            
        importances[col] = importances[col]/npermutations
    return importances

def train_based_feature_importance(Xtrain,ytrain,Xtest,ytest, npermutations = 20):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = get_accuracy(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtrain2 = Xtrain.copy()
            Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values
            new_accuracy = get_accuracy(Xtrain2, ytrain, Xtest, ytest)
            feature_importance = orig_accuracy - new_accuracy
            importances[col] += feature_importance
            
        importances[col] = importances[col]/npermutations
    return importances