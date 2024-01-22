import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes
from kneed import KneeLocator

from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Loading dataset
dataset = pd.read_csv('C:/Users/ulast/OneDrive/Masaüstü/shopping_behavior_updated.csv')

# Exclude Customer ID
dataset.drop(['Customer ID'], axis='columns', inplace=True)

dataset.isnull().sum()

# Display the outliers
df_cat = dataset
for i in df_cat.columns:
    print(df_cat[i].unique())

# Transform categorical labels to numerical labels
df_cat = dataset[['Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 'Season', 'Subscription Status',
                  'Shipping Type', 'Discount Applied', 'Promo Code Used', 'Payment Method', 'Frequency of Purchases']]
encoders = {}

for col_name in df_cat.columns:
    series = df_cat[col_name]
    label_encoder = LabelEncoder()
    df_cat[col_name] = pd.Series(label_encoder.fit_transform(series[series.notnull()]),
                                 index=series[series.notnull()].index)
    encoders[col_name] = label_encoder

# Handling missing values if there exist

df_num = dataset[['Age', 'Purchase Amount (USD)', 'Previous Purchases', 'Review Rating']]
imputer = KNNImputer(n_neighbors=5)
df_num.loc[:] = imputer.fit_transform(df_num)

imputer = KNNImputer(n_neighbors=1)
df_cat.loc[:] = imputer.fit_transform(df_cat)

dataset = pd.concat([df_cat, df_num], axis=1)

pca = PCA(2)
d_f = pca.fit_transform(dataset)
d_f.shape

# Decode the categorical data for the k-prototype
for i in df_cat.columns:
    dataset[i] = dataset[i].astype(int)
for col_name in df_cat.columns:
    dataset[col_name] = encoders[col_name].inverse_transform(dataset[col_name])

dataset.info()

# See the position of the categorical data
catColumnsPos = [dataset.columns.get_loc(col) for col in list(dataset.select_dtypes('object').columns)]

print('Categorical columns              : {}'.format(list(dataset.select_dtypes('object').columns)))
print('Categorical columns position     : {}'.format(catColumnsPos))

# Dataframe to matrix
dfMatrix = dataset.to_numpy()

# CLUSTERING

# Using elbow method to find the number of cluster
from kmodes.kprototypes import KPrototypes

cost = []
for cluster in range(1, 4):
    try:
        kpototype = KPrototypes(n_jobs=-1, n_clusters=cluster, init='Huang', random_state=0)
        kpototype.fit_predict(dfMatrix, categorical=catColumnsPos)
        cost.append(kpototype.cost_)
        print('Cluster initiation: {}'.format(cluster))
    except:
        break

plt.plot(cost)
plt.title('Cost over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# See the number of clusters with KneeLocator
from kneed import KneeLocator

cost_knee_c3 = KneeLocator(
    x=range(1, 4),
    y=cost,
    S=0.1, curve="convex", direction="decreasing", online=True)

K_cost_c3 = cost_knee_c3.elbow
print("elbow at k =", f'{K_cost_c3:.0f} clusters')

# Implementing the k-prototype
kpototype = KPrototypes(n_jobs=-1, n_clusters=2, init='Huang', random_state=0)
dataset['clusters'] = kpototype.fit_predict(dfMatrix, categorical=catColumnsPos)

label = kpototype.fit_predict(dfMatrix, categorical=catColumnsPos)
print(label)

# Visualizing the clusters
u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(d_f[label == i, 0], d_f[label == i, 1], label=i)
plt.legend()
plt.show()

# The volume of each cluster
dataset['clusters'].value_counts().plot(kind='bar')
plt.show()

cluster_colors = ['blue', 'orange']

# Stats of categorical data
for col_name in df_cat.columns:
    plt.figure(figsize=(10, 9))

    for i, cluster_label in enumerate(dataset['clusters'].unique()):
        subset = dataset[dataset['clusters'] == cluster_label]
        counts = subset[col_name].value_counts(normalize=True)

        counts.plot(kind='bar', label=f'Cluster {cluster_label}', alpha=0.7, color=cluster_colors[i])

    plt.title(f'Distribution of {col_name} in Each Cluster')
    plt.xlabel(col_name)
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()

# Stats of numerical data
for col_name in df_num.columns:
    plt.figure(figsize=(10, 9))

    for cluster_label in dataset['clusters'].unique():
        subset = dataset[dataset['clusters'] == cluster_label]
        sns.histplot(subset[col_name], label=f'Cluster {cluster_label}', kde=True, alpha=0.7)

    plt.title(f'Distribution of {col_name} in Each Cluster')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()






# K-modes based on categorical datas
'''
cost = []
for cluster in range(1, 11):  # 1'den 10'a kadar küme sayısını dene
    km = KModes(n_clusters=cluster, init='Huang', n_init=5, verbose=1)
    km.fit_predict(df_cat)
    cost.append(km.cost_)

# KneeLocator kullanarak uygun küme sayısını belirleyin
knee_locator = KneeLocator(range(1, 11), cost, curve='convex', direction='decreasing')
optimal_clusters = knee_locator.elbow

# Elbow methodunun grafiğini çizin
plt.plot(range(1, 11), cost, marker='o')
plt.xlabel('Küme Sayısı')
plt.ylabel('Toplam Maliyet')
plt.title('Elbow Method: Küme Sayısının Belirlenmesi')
plt.vlines(optimal_clusters, plt.ylim()[0], plt.ylim()[1], linestyles='dashed', colors='red', label='Optimal Küme Sayısı')
plt.legend()
plt.show()

print("Optimal Küme Sayısı:", optimal_clusters)






km = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1)
clusters = km.fit_predict(df_cat)

# Add cluster labels to the original dataset
dataset['clusters'] = clusters

# Visualizing the clusters
sns.scatterplot(x='Category', y='Location', hue='clusters', data=dataset, palette='viridis')
plt.title('K-Modes Clustering')
plt.show()

# The volume of each cluster
dataset['clusters'].value_counts().plot(kind='bar')
plt.title('Cluster Sizes')
plt.xlabel('Cluster')
plt.ylabel('Size')
plt.show()

# Stats of categorical data
for col_name in df_cat.columns:
    plt.figure(figsize=(10, 9))
    sns.countplot(x=col_name, hue='clusters', data=dataset, palette='viridis')
    plt.title(f'Distribution of {col_name} in Each Cluster')
    plt.xlabel(col_name)
    plt.ylabel('Count')
    plt.show()
'''
