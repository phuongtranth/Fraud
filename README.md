# [Machine Learning] Churn Prediction and Segmentation for Targeted Promotions
## I. Introduction
In this project, I analyzed churn behavior for an **e-commerce company** using the dataset about customer base to **identify patterns of churned users** and offer strategic **recommendations** for reducing customer churn. I built and **fine-tuned a Random Forest machine learning model** to predict user churn and applied **K-means clustering** to segment churned users into distinct groups. Based on these segments, I provided insights for **targeted promotions** to support the company's marketing and customer retention teams in designing data-driven strategies.
### Dataset
The dataset is attached above is dataset about customer base of an e-commerce company with the explaintion as below:
| Variable                     | Description                                                  |
| ---------------------------- | ------------------------------------------------------------ |
| CustomerID                   | Unique customer ID                                           |
| Churn                        | Churn Flag                                                   |
| Tenure                       | Tenure of customer in organization                           |
| PreferredLoginDevice         | Preferred login device of customer                           |
| CityTier                     | City tier (1,2,3): miền                                      |
| WarehouseToHome              | Distance in between warehouse to home of customer            |
| PreferPaymentMethod          | PreferredPaymentMode Preferred payment method of customer    |
| Gender                       | Gender of customer                                           |
| HourSpendOnApp               | Number of hours spend on mobile application or website       |
| NumberOfDeviceRegist ered    | Total number of devices is registered on particular customer |
| PreferedOrderCat             | Preferred order category of customer in last month           |
| SatisfactionScore            | Satisfactory score of customer on service                    |
| MaritalStatus                | Marital status of customer                                   |
| NumberOfAddress              | Total number of added added on particular customer           |
| Complain                     | Any complaint has been raised in last month                  |
| OrderAmountHikeFroml astYear | Percentage increases in order from last year                 |
| CouponUsed                   | Total number of coupon has been used in last month           |
| OrderCount                   | Total number of orders has been places in last month         |
| DaySinceLastOrder            | Day Since last order by customer                             |
| CashbackAmount               | Average cashback in last month                               |

## II. Exploring the patterns/behavior of churned users
### 1. Data Cleaning
#### Handle Missing/ Duplicates values
![image](https://github.com/user-attachments/assets/7634988e-2e40-4d31-bd78-50e47a5ade53)

- Tenure: replace by its median

- WarehouseToHome: replace by its median

- HourSpendOnApp: replace by its mean

- OrderAmountHikeFromlastYear: New customers with no previous year's data for comparison => replace by 0

- CouponUsed: Customers who haven't used any coupons => replace by 0

- OrderCount:  customers who haven't placed orders => replace by 0

- DaySinceLastOrder: can be converted to its median

![image](https://github.com/user-attachments/assets/41c67862-3d1e-4eff-9480-06331b4bed6a)

- No duplications

#### Unvariable Analyse
To perform univariate analysis, first I identify the numeric columns in the dataset using the data types of each column. Then, I count and print the number of unique values for each numeric column. Following this, I create boxplots for each numeric column using seaborn. These boxplots visually represent the distribution of each feature, showing the median, quartiles, and potential outliers. I will also use the describe() function to obtain specific statistical measures. 

This combined approach of descriptive statistics and visualizations provides a thorough initial examination of each numeric variable's characteristics in the dataset.

```python
# Numberic data:
numeric_cols = raw_data.loc[:, raw_data.dtypes != object].columns.tolist()


for col in numeric_cols:
    print(f"Unique values of {col}: {raw_data[col].nunique()}")

for col in numeric_cols:
    plt.figure(figsize=(10, 6))  # Set the size of the plot
    sns.boxplot(x=raw_data[col])
    plt.title(f'Boxplot of {col}')  # Title of the plot
    plt.show()

# Category data:

cate_cols = raw_data.loc[:, raw_data.dtypes == object].columns.tolist()

for col in cate_cols:
    print(f"Unique values of {col}: {raw_data[col].nunique()}")
```
```python
raw_data.describe()
```
- Tenure: right-skewed, most customers clustered in 0 - 20
- WarehouseToHome: majority of customers having short distances (below 40) with some outliers >120
- HourSpendOnApp: centered around 2-3 hours
- NumberOfDeviceRegistered: centered around 3-4 devices
- SatisfactionScore: mean at 3, suggesting generally moderate to positive customer satisfaction levels.
- NumberOfAddress : most customers have 1-5 addresses, several outliers having up to 22 addresses
-OrderAmountHikeFromlastYear: growth in customer spending, centered around 13-18%
- CouponUsed: most customers using 0-2 coupons, some outliers using up to 16
- OrderCount: most customers placing 1-3 orders and several outliers up to 16 orders
- DaySinceLastOrder: Most customers have ordered recently (within 0-10 days)
- CashbackAmount: cashback distribution is complex, cashback distribution is complex, a cluster of high outliers around 300
#### Outliers Detection
Remove the rows that have a outliers causing the data to be skewed:
- Tenuere: >40
- WarehouseToHome >100
- NumberOfAddress >15
- DaySinceLastOrder >20
```python
cleaned_data = raw_data[
    (raw_data['Tenure'] <= 40) &
    (raw_data['WarehouseToHome'] <= 100) &
    (raw_data['NumberOfAddress'] <= 15) &
    (raw_data['DaySinceLastOrder'] <= 20)
]

cleaned_data = cleaned_data.reset_index(drop=True)
```

### 2. Transform features
To transform categorical features into a format suitable for machine learning algorithms, I use one-hot encoding. This process converts categorical variables into binary (0 or 1) columns, making them usable in many ML models.
```python
cate_columns = cleaned_data.loc[:, cleaned_data.dtypes == object].columns.tolist()

encoded_df = pd.get_dummies(cleaned_data, columns = cate_columns,drop_first=True)
encoded_df.shape
```
### 3. Apply base Random Forest model
I first split the dataset into features (X) and target variable (y), excluding 'CustomerID'. The data is then divided into training and testing sets. I normalize the features using StandardScaler to ensure all variables are on the same scale. Finally, I initialize a Random Forest classifier with a maximum depth of 2, train it on the scaled training data, and use it to make predictions on the scaled test set. 

```python
# Split train/test sets
from sklearn.model_selection import train_test_split
X = encoded_df.drop(['Churn','CustomerID'], axis=1)
y = encoded_df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the features

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Random Forest
from sklearn.ensemble import RandomForestClassifier

clf_rand = RandomForestClassifier(max_depth=2, random_state=0)

clf_rand.fit(X_train_scaled, y_train)

y_ranf_pre_test = clf_rand.predict(X_test_scaled)
```

### 4. Show Feature Importance from model
I extract and visualize feature importance scores. This process involves ranking features based on their importance, creating a bar plot to visually represent these importances, and printing the top 20 most important features. 
```python
feats = {} # a dict to hold feature_name: feature_importance
for feature, importance in zip(X.columns, clf_rand.feature_importances_):
    feats[feature] = importance #add the name/value pair

importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
importances = importances.sort_values(by='Gini-importance', ascending=True)

importances = importances.reset_index()

# Create bar chart
plt.figure(figsize=(10, 10))
plt.barh(importances.tail(20)['index'][:20], importances.tail(20)['Gini-importance'])

plt.title('Feature Important')

# Show plot
plt.show()
```
![image](https://github.com/user-attachments/assets/3e34942b-fe2c-47d7-bcd4-083eee2f5885)

 Top 4 features choosed to analyse:

- Tenure

- Complain

- DaySinceLastOrder

- CashbackAmount

```python
def plot_distribution(feature):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=encoded_df, x=feature, hue='Churn', kde=True, element='step')
    plt.title(f'Distribution of {feature} by Churn Status')
    plt.show()

# Analyze each important feature
for feature in ['Tenure', 'Complain', 'DaySinceLastOrder', 'CashbackAmount']:
    plot_distribution(feature)

    # Print summary statistics
    print(f"\nSummary statistics for {feature}:")
    print(encoded_df.groupby('Churn')[feature].describe())

    # Correlation with Churn
    correlation = encoded_df['Churn'].corr(encoded_df[feature])
    print(f"\nCorrelation between {feature} and Churn: {correlation:.2f}")
```
To further investigate the top features identified by the Random Forest model, I create a custom function to visualize the distribution of each feature in relation to customer churn. This function plots histograms with kernel density estimates, segmented by churn status. 
For each feature, I generate the distribution plot, print summary statistics grouped by churn status, and calculate its correlation with churn.
![image](https://github.com/user-attachments/assets/05626176-64d3-401c-ae61-af6aa6736b10)
![image](https://github.com/user-attachments/assets/c3dcd880-c264-4693-8b73-0ac662af0484)
![image](https://github.com/user-attachments/assets/801c4f28-663b-41a0-a4f0-0c208e9ed5e0)
![image](https://github.com/user-attachments/assets/befe1c3a-765d-457b-b740-0a86fcaebf5b)
### 5. The Key Patterns of Churners and Recommendations:
- Higher churn rate for new customers (0-1 month tenure)
→ Recommendation: Implement a 30-day engagement plan with personalized recommendations and exclusive discounts.

- Customers who complain are more likely to churn 
→ Recommendation: Establish a follow-up process after resolving complaints to ensure satisfaction.

- Customers with recent orders have slightly higher churn risk
→ Recommendation: Implement a quick feedback mechanism after purchase/delivery and analyze this trend across customer segments.

- Low cashback amounts (100 - 160) associated with higher churn risk
→ Recommendation: Introduce a tiered cashback system rewarding higher spending with increased cashback percentages.
## III. ML model for predicting churned users (fine tuning)
To improve my churn prediction model, I implement a more sophisticated approach using the top four features identified earlier. I then set up a Random Forest classifier and perform **hyperparameter tuning** using **GridSearchCV**. This exhaustive search over specified parameter values for the Random Forest model aims to find the combination that yields the best performance.
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(clf_rand, param_grid, cv=5, scoring='balanced_accuracy')
grid_search.fit(X_train_scaled, y_train)

# Print the best parameters
print("Best Parameters: ", grid_search.best_params_)

# Evaluate the best model on the test set
best_clf = grid_search.best_estimator_
accuracy = best_clf.score(X_test_scaled, y_test)
print("Test set accuracy: ", accuracy)
```
![image](https://github.com/user-attachments/assets/3e2e63ce-787e-402d-8cc6-b16e2750226b)

After finding the best hyperparameters I apply the optimized Random Forest model to make predictions on the test set. To evaluate the model's performance comprehensively, I generate a confusion matrix and calculate key classification metrics like precision, recall, and F1-score. 
![image](https://github.com/user-attachments/assets/6f743fcf-66bc-4526-98d8-9cc8f61a2794)

## IV. ML model for segmenting churned users
### 1. Dimension Reduction
To develop data-driven recommendations for targeted promotions, I focus on analyzing behavioral patterns among churners. I isolate the data for churned users and concentrate on four key features: Tenure, Complain, DaySinceLastOrder, and CashbackAmount.   
```python
churn_data = encoded_df[encoded_df['Churn']==1]
columns = ['Tenure', 'Complain', 'DaySinceLastOrder', 'CashbackAmount']
segmenting_data = churn_data[columns]
segmenting_data.info()
```
![image](https://github.com/user-attachments/assets/0e5b70f7-1092-4721-b69b-82d03f02633f)

Next, I normalize the selected features using MinMaxScaler. This ensures all features are on a comparable scale. I then apply Principal Component Analysis (PCA) to reduce the dimensionality of the data while preserving its most important characteristics. 

```python
from sklearn.preprocessing import MinMaxScaler

maxmin_scaler = MinMaxScaler()
X = maxmin_scaler.fit_transform(segmenting_data)

from sklearn.decomposition import PCA

pca = PCA()
pca_data = pca.fit_transform(X)

features = range(pca.n_components_)

plt.figure(figsize=(10, 6))
plt.bar(features, pca.explained_variance_ratio_)
plt.xticks(features)
plt.ylabel('Explained Variance Ratio')
plt.xlabel('PCA Feature')
plt.title('Explained Variance Ratio by PCA Feature')
plt.show()
```
![image](https://github.com/user-attachments/assets/c9b9fe25-66b0-42b6-880e-23be0a7fc659)

There's a noticeable drop-off in explained variance after the second component (Elbow Point). So, choosing the first two PCA feature (PC0, PC1) for building model.

### 2. Model Training
To determine the optimal number of clusters for segmenting our churned customers, I employ the Elbow Method using K-means clustering. 
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

ks = range(1, 11)
inertias = []

for k in ks:
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(df_reduced)
    inertias.append(kmeans.inertia_)

# Plot ks vs inertias
plt.plot(ks, inertias, '-o')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()
```
![image](https://github.com/user-attachments/assets/e476a1b6-07d2-431d-ac00-71caa5e2913f)

The "elbow" in this plot occurs around k=4, where the rate of decrease in inertia begins to level off significantly. 

I now apply the K-means Cluster algorithm with 4 cluster to previous reduced dataset.I also add these cluster labels to the data, allowing analyze each segment separately.
```python
kmeans = KMeans(n_clusters=4)
cluster_labels = kmeans.fit_predict(df_reduced)

# Add cluster labels to the DataFrame
df_reduced['Cluster'] = cluster_labels

# Calculate silhouette score
silhouette_avg = silhouette_score(df_reduced.drop('Cluster', axis=1), cluster_labels)
print(f"The average silhouette score is: {silhouette_avg:.3f}")
```
![image](https://github.com/user-attachments/assets/3677d70e-8366-4724-9334-958c64caef25)

Then, I calculate the mean values of each feature for each cluster. By examining these cluster means, we can identify distinct patterns and behaviors within each group of churned customers.
```python
segmenting_data['Cluster'] = cluster_labels

cluster_means = segmenting_data.groupby('Cluster').mean()
print("Cluster means for original features:")
print(cluster_means)
```
![image](https://github.com/user-attachments/assets/8e4d78b2-09b8-4e34-a58c-27d728c7a9be)

### 3. Segmentation and Recommendations for Promotion:
Based on the clustering data provided for churned users, we can identify 4 distinct clusters.
- Cluster 0: New, Satisfied, but Low-Value Customers
- Cluster 1: New, Dissatisfied Customers
- Cluster 2: Established, Satisfied, High-Value Customers
- Cluster 3: Established, Dissatisfied, High-Value Customers

Recommended actions for each group:
- Cluster 0: Offer a "loyalty boost" promotion and educational content about the platform.
- Cluster 1: Send a sorry email with a discount for the next purchase
- Cluster 2: Offer exclusive deals on their favorite categories, and add benefits to the customer program.
- Cluster 3: Offer personalized customer service to address their concerns and exclusive deals on their favorite categories.

