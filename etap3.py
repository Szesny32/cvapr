!pip install umap-learn
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

success_fail_data = df[df['state'].isin(['successful', 'failed'])]

features = ['backers_count', 'usd_pledged', 'goal']
X = success_fail_data[features]

pca = PCA(n_components=2)  # Tworzymy instancję klasy PCA z 2 składowymi głównymi
X_pca = pca.fit_transform(X)

pca_df = pd.DataFrame(data=X_pca, columns=['PCA1', 'PCA2'])

plt.scatter(pca_df['PCA1'], pca_df['PCA2'])
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA Analysis')
plt.show()

##

reducer = umap.UMAP(n_components=2)
X_umap = reducer.fit_transform(X)
umap_df = pd.DataFrame(data=X_umap, columns=['UMAP1', 'UMAP2'])
X_train, X_test, y_train, y_test = train_test_split(umap_df, success_fail_data['state'], test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


