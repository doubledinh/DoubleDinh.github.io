---
layout: post
title: Classification Project
---
#### Predicting the for-profit status of a given college

This classification project revolved around predicting which colleges from a given dataset  were for-profit. The dataset was then constricted to a set number of categories that applied to the colleges' for-profit status. From there, I used a method called selectKBest which lists the features that have the most statistically significant relationships with the colleges' status. What I found was that the following had the closest relationships: average family income, branches, faculty salary, instructional expenditure, loan principal, pell grant debt, size, and tuition revenue. Based on the results, I believe that families who earn a lot of money and thus are more willing to dish out more money to earn a degree for their children most likely put their children into for-profit schools. In other words, more affluent families would almost always pay their way to have their kids receive a degree. Interestingly, however, the accreditation status of a college does not influence their for-profit status. I believe this is due to the fact that families who are willing to pay disregard whether the school is creditable or not. 

After I determined the most significant categories, I ran test/train split and fit the training set into a logistic regression model. After running the model, I found the both the training and testing accuracy to be 0.90, which is a rather high number. I then found that the optimal number of nearest neighbors is 5 using a grid search. With the model set to 5 nearest neighbors, the accuracy was .89, which is slightly lower than the logistic model. I then printed a classification report of the KNN model and described what the terms mean below: 

Precision: Out of all the colleges that I predicted as being for profit, I was correct 90% of the time. Contrarily, with a precision score of 94%, out of all the colleges that I predicted as being not for profit, we were correct 94% of the time.

Recall: With a recall score of 92%, out of all colleges that were not for profit, we found 92% of them. Whereas, out of all the colleges that were for profit, we found 92% of them. 

Accuracy: We were correctly predicted 92% of all the colleges in the test set. 

Support: There were 3835 not for profit colleges and 2905 for profit colleges.

F1 Score: This is the balance between the recall and the precision of the model.

I then printed a confusion matrix using sklearn, however, I discovered that my computer was unable to correctly display the elegant diagram. As a result, I printed out the bare numbers and made the following conclusions:

Out of all the colleges labeled as not non profit, we correctly predicted 3525 colleges and incorectly predicted 310. By contrast, out of all the colleges labeled as for profit, we correctly identified 2671, and incorectly predicted 234.




You can find my code on my [repository](https://github.com/doubledinh/Classification_Project) or below.

#### Classification Project

```python
df = pd.read_csv('data/schools.csv', index_col = 0)
print(df.shape)
```

    (6740, 46)

We are going to use classification algorithms to try to predict for-profit or non-profit status. 

Some of the categories aren't applicable to what we want to do so let's consider the following columns only:



```python
X = df[['size','retention','branches', 'online_only', 'under_investigation', 'most_common_degree', 'highest_degree',
       'faculty_salary', 'instructional_expenditure_per_fte',
       'tuition_revenue_per_fte', 'part_time_share',
       'age_entry', 'percent_dependent', 'first_generation', 'percent_black',
        'avg_family_income','ind_low_income', 'dep_low_income', 'loan_principal',
       'federal_loan_rate', 'students_with_any_loans',
       'pell_grant_debt', 'percent_pell_grant',
       'fafsa_sent', '7_yr_repayment_completion', '5_year_declining_balance',
       'relig_y_n', 'accred_y_n', 'retention_listed_y_n',
       'fac_salary_listed_y_n', '7_yr_repayment_completion_y_n',
       '5_year_declining_balance_y_n', 'for_profit']]
```

Let's first use a method called [SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html) to see which features have the most statistically significant relationships with profit status. The **lower** the p value, the **more** statistically significant:


```python
y = X.pop('for_profit')

X_new = SelectKBest(chi2, k=2).fit(X, y)

features = []
for i, column in enumerate(X.columns):
    features.append((X_new.pvalues_[i], column))
features.sort()
    
    
print('chi2-value', 'feature')
features
```

    chi2-value feature
    [(0.0, 'avg_family_income'),
     (0.0, 'branches'),
     (0.0, 'faculty_salary'),
     (0.0, 'instructional_expenditure_per_fte'),
     (0.0, 'loan_principal'),
     (0.0, 'pell_grant_debt'),
     (0.0, 'size'),
     (0.0, 'tuition_revenue_per_fte'),
     (5.339096890496358e-283, 'highest_degree'),
     (2.3550018048916394e-222, 'most_common_degree'),
     (5.4025502611485345e-149, 'fac_salary_listed_y_n'),
     (8.240679129762035e-148, 'relig_y_n'),
     (5.293172158552389e-143, 'age_entry'),
     (1.412516031160619e-48, 'percent_dependent'),
     (1.429381130658714e-24, '5_year_declining_balance'),
     (1.3527931483014148e-21, 'federal_loan_rate'),
     (6.095800240149156e-17, 'percent_black'),
     (3.2497574845087588e-15, 'fafsa_sent'),
     (2.3099399670295238e-10, '7_yr_repayment_completion'),
     (5.33226165965135e-09, 'first_generation'),
     (1.4079344574521428e-08, 'part_time_share'),
     (1.51637032044754e-08, 'dep_low_income'),
     (5.3899600324359155e-08, 'percent_pell_grant'),
     (2.942362811776858e-06, 'students_with_any_loans'),
     (9.035892891499908e-06, 'under_investigation'),
     (0.006314439338573356, 'retention_listed_y_n'),
     (0.012536378966747178, 'online_only'),
     (0.01746153785302236, 'ind_low_income'),
     (0.5021649291685133, '5_year_declining_balance_y_n'),
     (0.6158853388832435, 'retention'),
     (0.8250416943743915, 'accred_y_n'),
     (0.9230148052834302, '7_yr_repayment_completion_y_n')]



1.Based on the info above, what was intuitive? What was surprising? How big (or small) of an effect does accredition have on for-profit status? What might be some guesses as to why this is the case?

#### How big (or small) of an effect does accredition have on for-profit status? 
Accredation has very little effect on whether the college is for profit.

#### What was intuitive?
I assume that families who earn a lot and are willing to dish out a lot of money to earn a degree for their children most likely put their children into for profit schools. Essentially, rich families would almost always pay their way to a degree. 

#### What might be some guesses as to why this is the case?
Regardless of whether the school is creditable or not, parents would still pay for a degree.

2.Do a test/train split and give the testing accuracy error for logistic regression.


```python
#insert 2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model_log = LogisticRegression(multi_class = "auto", solver = 'lbfgs', max_iter=10000)
model_log.fit(X_train, y_train)

print('train accuracy ', model_log.score(X_train, y_train))
print('test accuracy', model_log.score(X_test,y_test))
```

    train accuracy  0.8986858838490887
    test accuracy 0.8966369930761622


3.Find the optimal number of nearest neighbors for KNN using grid search and then give the test accuracy. 


```python
#insert 3
param_grid = {'n_neighbors': range(1, 20)}

model_Grid = GridSearchCV(KNeighborsClassifier(), 
                    param_grid, 
                    cv=3, 
                    scoring='accuracy')

model_Grid = model_Grid.fit(X_train, y_train)

print(model_Grid.best_params_, model_Grid.score(X_test,y_test))
```

    {'n_neighbors': 5} 0.8921859545004945


4.Run a classification report and describe in detail what the terms mean in the context of your model.


```python
#insert 4
print(classification_report(y, model_Grid.predict(X)))
```

                  precision    recall  f1-score   support
    
               0       0.93      0.91      0.92      3835
               1       0.89      0.91      0.90      2905
    
        accuracy                           0.91      6740
       macro avg       0.91      0.91      0.91      6740
    weighted avg       0.91      0.91      0.91      6740
    


Precision: Out of all the colleges that I predicted as being for profit, I was correct 90% of the time. Contrarily, with a precision score of 94%, out of all the colleges that I predicted as being not for profit, we were correct 94% of the time.

Recall: With a recall score of 92%, out of all colleges that were not for profit, we found 92% of them. Whereas, out of all the colleges that were for profit, we found 92% of them. 

Accuracy: We were correctly predicted 92% of all the colleges in the test set. 

Support: There were 3835 not for profit colleges and 2905 for profit colleges.

F1 Score: This is the balance between the recall and the precision of the model.

5.Print a confusion matrix and describe what it means in your context.


```python
# Generate a confusion matrix plot: 

def plot_confusion_matrix(cm, classes=[0, 1], title='some confusion matrix',
                          normalize=False,
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center", size=20,
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
plot_confusion_matrix(confusion_matrix(y, model_Grid.predict(X)))

confusion_matrix(y, model_Grid.predict(X))
```




    array([[3505,  330],
           [ 259, 2646]])




<img src="/images/Confusion.png"/>


Out of all the colleges labeled as not non profit, we correctly predicted 3525 colleges and incorectly predicted 310. By contrast, out of all the colleges labeled as for profit, we correctly identified 2671, and incorectly predicted 234.

6.Make a comparative ROC plot of the naive bayes, logistic, gradient boosting, and KNN classifiers.


```python
#insert 6
def plot_roc(ytrue, yproba, model, title='some ROC curve'):
    auc = roc_auc_score(ytrue, yproba)
    fpr, tpr, thr = roc_curve(ytrue, yproba)
    plt.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=.4)
    plt.plot(fpr, tpr, label='{} auc={:.2f}%'.format(model, auc*100))
    plt.axis('equal')
    plt.xlim([-.02, 1.02])
    plt.ylim([-.02, 1.02])
    plt.ylabel('tpr')
    plt.xlabel('fpr')
    plt.title(title)
    plt.legend(loc='best')
    plt.grid(True)
    
model = naive_bayes.GaussianNB()
model.fit(X_train, y_train)

model_grad = GradientBoostingClassifier()
model_grad.fit(X_train, y_train)
   
y_proba_NB = model.predict_proba(X_test)[:, 1]
y_proba_KN = model_Grid.predict_proba(X_test)[:, 1]
y_proba_LR = model_log.predict_proba(X_test)[:, 1]
y_proba_GB = model_grad.predict_proba(X_test)[:, 1]

plot_roc(y_test, y_proba_NB, "Naive Bayes")
plot_roc(y_test, y_proba_KN, "K Nearest Neighbors")
plot_roc(y_test, y_proba_LR, "Logistic Regression")
plot_roc(y_test, y_proba_GB, "Gradient Boosting")
```


<img src="/images/ROC1.png"/>

7. Using the logarithmic regression model, plot a decision boundary between instructional_expenditure_per_fte and 5_year_declining_balance. Does it appear that for-profit status has a clear boundary based on these predictors?


```python
X = df[['instructional_expenditure_per_fte', '5_year_declining_balance']]
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)
X = pd.DataFrame(X)

Q = X.values
h = .02  # meshsize
x_min, x_max = Q[:, 0].min() - .5, Q[:, 0].max() + .5 
y_min, y_max = Q[:, 1].min() - .5, Q[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
model.fit(X.iloc[:,:2], y)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]) # ravel() flattens the data

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(15, 10))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(Q[:, 0], Q[:, 1], c=y, edgecolors='k', cmap=plt.cm.Paired)
plt.title('Decision boundaries with 2 attributes')
plt.xlabel('instructional expenditure')
plt.ylabel('5 year declining balance')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
```




    (-2.941558886777559, 3.1784411132224464)




![png](ClassificationProject_files/ClassificationProject_22_1.png)


Looking at the color gradients, it appears as if is a horizontal boundary when the 5 year declining balance is at 0. However, based on the example we did in class, the data points are quite merged together.

8. We have not covered random forests but they are a very popular type of classifier. It is very good practice in reading the docs to get a new classifier working. Read [this](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) and then apply the RandomForestClassifier().


```python
#insert 8
model_clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
model_clf.fit(X_train, y_train)  
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=2, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=100,
                           n_jobs=None, oob_score=False, random_state=0, verbose=0,
                           warm_start=False)



9. Support vector machines are another type of classifier. Read the docs [here](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and then try implementing that one.


```python
from sklearn.svm import SVC
X_train = X_train.sort_index().reset_index(drop = True)
X_train_Q = X_train.drop(columns = ['branches', 'online_only', 'under_investigation', 'relig_y_n', 'accred_y_n', 'retention_listed_y_n','fac_salary_listed_y_n', '7_yr_repayment_completion_y_n','5_year_declining_balance_y_n'])
X_train_C = X_train[['branches', 'online_only', 'under_investigation', 'relig_y_n', 'accred_y_n', 'retention_listed_y_n','fac_salary_listed_y_n', '7_yr_repayment_completion_y_n','5_year_declining_balance_y_n']]

Scaler = StandardScaler()
Scaler.fit(X_train_Q)
X_train_Q = Scaler.transform(X_train_Q)
X_train_Q = pd.DataFrame(X_train_Q)

X_train_Scaled = X_train_C.join(X_train_Q)
X_train_Scaled

model_SVC = SVC(gamma = 'auto', probability = True)
model_SVC.fit(X_train_Scaled, y_train) 
```




    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
        max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
        verbose=False)



10. Make a comparative ROC plot of the naive bayes, logistic, gradient boosting, KNN classifiers, random forest, and SVM classifiers.


```python
y_proba_RF = model_clf.predict_proba(X_test)[:, 1]
y_proba_SVC = model_SVC.predict_proba(X_test)[:, 1]

plot_roc(y_test, y_proba_SVC, "Support Vector Machines")
plot_roc(y_test, y_proba_RF, "Random Forest")
plot_roc(y_test, y_proba_NB, "Naive Bayes")
plot_roc(y_test, y_proba_KN, "K Nearest Neighbors")
plot_roc(y_test, y_proba_LR, "Logistic Regression")
plot_roc(y_test, y_proba_GB, "Gradient Boosting")
```


![png](ClassificationProject_files/ClassificationProject_29_0.png)


11. Take in a new school's data and predict the school's profit status using one of your previous classifier methods.


```python
#insert 11
print(model_grad.predict([X_test.iloc[1]]))
```

    [0]


12. What are the worst schools? Some of them are chains of schools so don't necessarily consider singular names but generalize to chains. Give a lot of justification for your analysis as everyone might have a different answer here. Insert these responses into your blog.


```python
#insert 12
Worst_Cate = df[['loan_principal', 'students_with_any_loans', '7_yr_repayment_completion', '5_year_declining_balance', 'retention']]
Worst_Cate_Transform = Worst_Cate.assign(seven_yr_repayment_nocompletion = lambda x: 1 - Worst_Cate['7_yr_repayment_completion'])
Worst_Cate_Transform = Worst_Cate_Transform.assign(five_year_increasing_balance = lambda x: 1 - Worst_Cate['5_year_declining_balance'])
Worst_Cate_Transform = Worst_Cate_Transform.assign(retentionrate = lambda x: 1 - Worst_Cate['retention'])
Worst_Cate_Transform = Worst_Cate_Transform.drop(columns = ['7_yr_repayment_completion', '5_year_declining_balance', 'retention'])

name = df['name']
name = name.sort_index().reset_index(drop = True)
name = pd.DataFrame(name)

Scalar = StandardScaler()
Worst = Scalar.fit_transform(Worst_Cate_Transform)
Worst = pd.DataFrame(Worst)

Worst = Worst.assign(Score = lambda x: Worst[0] + Worst[1] + Worst[2] + Worst[3] + Worst[4])
Worst = name.join(Worst)
Worst = Worst.drop(columns = [0,1,2,3,4])
Worst = Worst.sort_values(by='Score', ascending = False)
Worst.head(11)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1889</td>
      <td>Stevens-Henager College</td>
      <td>10.371879</td>
    </tr>
    <tr>
      <td>1678</td>
      <td>Beulah Heights University</td>
      <td>9.381801</td>
    </tr>
    <tr>
      <td>2065</td>
      <td>Sanford-Brown College-Chicago</td>
      <td>8.855680</td>
    </tr>
    <tr>
      <td>4538</td>
      <td>National College-Willoughby Hills</td>
      <td>8.480454</td>
    </tr>
    <tr>
      <td>5737</td>
      <td>National College-Nashville</td>
      <td>8.373743</td>
    </tr>
    <tr>
      <td>2524</td>
      <td>Virginia College-Shreveport Bossier City</td>
      <td>8.141511</td>
    </tr>
    <tr>
      <td>196</td>
      <td>CollegeAmerica-Phoenix</td>
      <td>7.961229</td>
    </tr>
    <tr>
      <td>274</td>
      <td>CollegeAmerica-Flagstaff</td>
      <td>7.961229</td>
    </tr>
    <tr>
      <td>5723</td>
      <td>Lane College</td>
      <td>7.935538</td>
    </tr>
    <tr>
      <td>5475</td>
      <td>Kenneth Shuler School of Cosmetology-Columbia</td>
      <td>7.847874</td>
    </tr>
    <tr>
      <td>5528</td>
      <td>Allen University</td>
      <td>7.713083</td>
    </tr>
  </tbody>
</table>
</div>



13. If you were a governmental organization overseeing accreditation, what factors would be most important to you in making sure that the college was non-predatory? Give a lot of detail here as well in your blog.

When I researched into colleges that were predatory, I found that the most common characteristics between the collges were their high student debt. At times, the student debt would reach a value that the students could never repay. In the case of our dataset, the categorical variable that we should look at is the percentage of students that successfully repays the college after 7 years. Furthermore, I would look at the percentage of students with student loans and compare that to the loan principal. If the college has a high percentage of students currently in dept via student loans, has a high loan principal, and has a low percntage of students that successfully repays the college after 7 years, I would classify that college as predatory. 

14.Read several articles on college predatory practices and cite and incorporate them into your blog discussion. Remember to link to them clearly by using the 
```[here](http://....)``` syntax.

```python
#https://bigthink.com/politics-current-affairs/predatory-student-loans
```
