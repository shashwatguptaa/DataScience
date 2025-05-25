import numpy as np
from collections import Counter
class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature=feature
        self.threshold=threshold
        self.left=left
        self.right=right
        self.value=value

    def isLeafNode(self):
        return self.value is not None
    
class DecisionTree:
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split=min_samples_split
        self.max_depth=max_depth
        self.n_features=n_features
        self.root=None
    
    def fit(self,X,y):
        self.n_features=X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root=self.grow_tree(X,y)

    def grow_tree(self,X,y,depth=0):
        n_samples,n_feats=X.shape
        n_labels=len(np.unique(y))
        if(depth>=self.max_depth or n_labels==1 or n_samples<=self.min_samples_split):
            leaf_value=self.most_common_labels(y)
            return Node(value=leaf_value)

        feat_idxs=np.random.choice(n_feats,self.n_features,replace=False)
        best_thresh,best_feature=self.best_split(X,y,feat_idxs)

        left_indxs,right_indxs=self.split(X[:,best_feature],best_thresh)
        left=self.grow_tree(X[left_indxs,:],y[left_indxs],depth+1)
        right=self.grow_tree(X[right_indxs,:],y[right_indxs],depth+1)

        return Node(best_feature,best_thresh,left,right)

    def best_split(self,X,y,feat_idxs):
        best_gain=-1
        split_idx,split_thresh=None,None

        for feat_idx in feat_idxs:
            X_column=X[:,feat_idx]
            threshholds=np.unique(X_column)
            for threshold in threshholds:
                gain=self.information_gain(X_column,y,threshold)
                if gain > best_gain:
                    best_gain=gain
                    split_idx=feat_idx
                    split_thresh=threshold

        return split_thresh,split_idx
    
    def information_gain(self,X_column,y,threshold):
        parent_entropy=self.entropy(y)
        left_indxs,right_indxs=self.split(X_column,threshold)
        if len(left_indxs)==0 or len(right_indxs)==0:
            return 0
        
        n=len(y)
        n_l, n_r=len(left_indxs),len(right_indxs)
        e_l,e_r=self.entropy(y[left_indxs]),self.entropy(y[right_indxs])
        child_entropy=(n_l/n)*e_l+(n_r/n)*e_r
        info_gain=parent_entropy-child_entropy
        return info_gain

    def split(self,X_column,threshold):
        left_indxs=np.argwhere(X_column <= threshold).flatten()
        right_indxs=np.argwhere(X_column > threshold).flatten()

        return left_indxs,right_indxs
    
    def entropy(self,y):
        hist =np.bincount(y)
        ps=hist/len(y)
        return -np.sum([p*np.log(p) for p in ps if p>0])


    def most_common_labels(self,y):
        counter=Counter(y)
        value=counter.most_common(1)[0][0]
        return value
    
    def predict(self,X):
        return np.array([self.traverse_tree(x,self.root) for x in X])

    def traverse_tree(self,x,node):
        if(node.isLeafNode()):
            return node.value
        
        if(x[node.feature]<=node.threshold):
            return self.traverse_tree(x,node.left)
        return self.traverse_tree(x,node.right)
