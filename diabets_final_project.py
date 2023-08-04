
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score




pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_csv(r"C:\Users\user\Desktop\Datas\diabetes.csv")
df = df_.copy()
df.head(30)
df.info()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
       num_col = [col for col in dataframe.columns if dataframe[col].nunique() >= cat_th and
                  dataframe[col].dtypes != "O"]
       num_but_cat_col = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                        dataframe[col].dtypes != "O"]

       cat_but_car_col = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                          dataframe[col].nunique() >= car_th]

       cat_col = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and
                          dataframe[col].nunique() < car_th]

       cat_col =cat_col + num_but_cat_col

       return num_col, cat_col, cat_but_car_col


num_col, cat_col, cat_but_car_col = grab_col_names(df)

for col in num_col:
       print(df.groupby("Outcome")[col].mean(),col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_col:
    num_summary(df, col, plot=True)



def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_col:
    target_summary_with_num(df, "Outcome", col)

correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()


y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.77
# Recall: 0.706 # pozitif sınıfın ne kadar başarılı tahmin edildiği
# Precision: 0.59 # Pozitif sınıf olarak tahmin edilen değerlerin başarısı
# F1: 0.64
# Auc: 0.75


def NULL(dataframe,col_name):
    print(dataframe[col_name], dataframe[col_name].isnull().sum())

for i in [df.columns]:
   print(i,NULL(df,i))

zero_cols = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]

def fill_NAN(dataframe, col_name):
    dataframe[col_name] = dataframe[col_name].replace(0, pd.NA)
    dataframe[col_name].fillna(dataframe[col_name].mean(), inplace=True)


for col in zero_cols:
    fill_NAN(df, col)



def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)



def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for i in num_col:
   print(i,outlier_thresholds(df,i))

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for i in num_col:
   print( i,check_outlier(df,i))


def grab_outliers(dataframe,col_name):
    low_limit, up_limit = outlier_thresholds(dataframe,col_name)
    print(dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)])

for i in num_col:
   print(i,grab_outliers(df,i))


def replace_with_tresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[dataframe[variable] < low_limit, variable] = int(low_limit)
    dataframe.loc[dataframe[variable] > up_limit, variable] = int(up_limit)

for i in num_col:
   replace_with_tresholds(df,i)

for col in df.columns:
    sns.boxplot(df[col])
    plt.title(f"{col}")
    plt.show(block=True)

df["Age_Class"] = pd.cut(df["Age"],[0,15,30,50,70,90],labels=["child","young","adult","measure","senior"])
df["BloodPressure_Class"] = pd.cut(x=df["BloodPressure"],bins=[0,70,90,120],labels=["low_pressure","normal_pressure","high_pressure"])
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])
df["NEW_GLUCOSE"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"])


df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"
df["was_pregnant"] = df["Pregnancies"].apply(lambda x: 1 if x>0 else 0 )
df.head()


num_col, cat_col, cat_but_car_col = grab_col_names(df)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols

for col in binary_cols:
    df = label_encoder(df, col)

cat_col = [col for col in cat_col if col not in binary_cols and col not in ["Outcome"]]

def one_hot_encoder(dataframe,col_name,drop_first = True):
    dataframe = pd.get_dummies(dataframe, columns=col_name, drop_first=drop_first)
    return dataframe
df= one_hot_encoder(df,cat_col)


def standart_scaler(dataframe,col_name):
    ss = StandardScaler()
    dataframe[col_name] = ss.fit_transform(dataframe[[col_name]])
    df.head()
for i in num_col:
    standart_scaler(df,i)

y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")



