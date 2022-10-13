from numpy import array, argmin
from sklearn.cluster import KMeans

# https://towardsdatascience.com/an-approach-for-choosing-number-of-clusters-for-k-means-c28e614ecb2c

class KmeansInertia:

    def __init__(self, inertia=0.1, n_clusters_max=5):
        self.inertia_ = inertia
        self.model = None
        self.n_clusters_ = n_clusters_max

    def fit(self, X, y=None, sample_weight=None):
        print(X)
        XX = array([x for x in X if all(x)])
        inertia_0 = ((XX - XX.mean(axis=0))**2).sum()
        model_list = []
        inertia_list = []
        for k in range(1, self.n_clusters_):
            kmeans = KMeans(k).fit(XX, sample_weight=sample_weight)
            model_list.append(kmeans)
            inertia_list.append(kmeans.inertia_ / inertia_0 + self.inertia_ * k)
            model_list.append(kmeans)
        best_k = argmin(inertia_list)
        self.model = model_list[best_k]
        return self

    def predict(self, X):
        return self.model.predict(X)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def get_parameters(self):
        params = {}
        params["model"] = self.model
        params["model_params"] = self.model.get_params()
        params["inertia_"] = self.inertia_
        params["cluster_centers_"] = self.model.cluster_centers_
        return params

    def set_parameters(self, params):
        self.model = params["model"]
        self.model.set_params(**params["model_params"])
        self.inertia_ = params["inertia_"]
        self.model.cluster_centers_ = params["cluster_centers_"]


if __name__ == '__main__':
    X = [[1, 2, 3 ,4], [2,1,4,3],[1, 2, 3 ,1], [2,1,4,1],[1, 2, 3 ,4], [2,2,4,3]]
    kmeans_ = KmeansInertia(inertia=0.1)
    print(kmeans_.fit_predict(X))
    print(kmeans_.get_parameters())
