from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def lda_method(train_data, train_labels, test_data, n_components):
    lda = LDA(n_components=n_components)
    train_lda = lda.fit_transform(train_data, train_labels)
    test_lda = lda.transform(test_data)
    return train_lda, test_lda

def ica_method(train_data, test_data, n_components):
    ica = FastICA(n_components=n_components)
    train_ica = ica.fit_transform(train_data)
    test_ica = ica.transform(test_data)
    return train_ica, test_ica

def tsne_method(train_data, test_data, n_components):
    tsne = TSNE(n_components=n_components)
    train_tsne = tsne.fit_transform(train_data)
    test_tsne = tsne.fit_transform(test_data)
    return train_tsne, test_tsne

def knn_method(train_data, train_labels, test_data, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    return knn.predict(test_data)

def logistic_regression_method(train_data, train_labels, test_data):
    log_r = LogisticRegression(max_iter=1000)
    log_r.fit(train_data, train_labels)
    return log_r.predict(test_data)

