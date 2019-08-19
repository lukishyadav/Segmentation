import pandas as pd
import my_module
import datetime
import seaborn as sns

threshold=60


df=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/Customer_Churn_Verification/march19/Churndaysfromlastrentalm19_modified_2019-7-30_1509.csv')


fmt = '%Y-%m-%d %H:%M:%S'
then = datetime.datetime.strptime('2019-03-31 23:59:59', fmt)

df['day of last rental '].iloc[1][0:19]

df['day of last rental ']=df['day of last rental '].apply(lambda x:datetime.datetime.strptime(x[0:19], fmt))


df['day of last rental ']=df['day of last rental '].apply(lambda x:then-x)


df['day of last rental ']=pd.to_timedelta(df['day of last rental '])

def convert(x):
    return x.total_seconds()/(3600*24)

df['d_f_l_r']=df['day of last rental '].apply(convert)


len(df[df['d_f_l_r']<0])

df=df[df['d_f_l_r']>0]

DF=df[df['d_f_l_r']>threshold]


#rental_count=pd.read_csv('/Users/lukishyadav/Desktop/Gittable_Work/Customer_Churn_Verification/RentalCountwithFilter_2019-7-9_1530.csv')


rental_count=pd.read_csv('/Users/lukishyadav/Desktop/segmentation/Customer_Churn_Verification/march19/Churnrentalcountm19_2019-7-30_1320.csv')
rental_count.columns=['customer_id', 'rental_count_2019']

Data4=pd.merge(DF[['customer_id', 'd_f_l_r']],rental_count[['customer_id','rental_count_2019']], on='customer_id', how='inner')
    
print(len(DF))
print(len(Data4))
print(1-len(Data4)/len(DF))


unchurned=set(DF['customer_id']) ^ set(Data4['customer_id'])

churned=set(Data4['customer_id'])

len(unchurned)
len(churned)