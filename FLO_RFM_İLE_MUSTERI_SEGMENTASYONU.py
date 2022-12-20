# RFM Analizi ile Müşteri Segmentasyonu Projesi #
# FLO#

##########################################
## TASK 1:  Veriyi Anlama ve Hazırlama ##
##########################################

#TASK 1.1:flo_data_20K.csv verisini okutup dataframe’in kopyası oluşturulur.

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
df_ = pd.read_csv("FLOMusteriSegmentasyonu/flo_data_20k.csv")
df = df_.copy

#TASK 1.2:Veri setinde a. İlk 10 gözlem, b. Değişken isimleri, c. Betimsel istatistik, d. Boş değer, e. Değişken tipleri, incelemesi yapınız.

df_.head(10)

df_.columns
 #Index(['master_id', 'order_channel', 'last_order_channel', 'first_order_date',
       #'last_order_date', 'last_order_date_online', 'last_order_date_offline',
      # 'order_num_total_ever_online', 'order_num_total_ever_offline',
       #'customer_value_total_ever_offline', 'customer_value_total_ever_online',
       #'interested_in_categories_12'],
     # dtype='object')
df_.describe().T

df_.isnull().sum()

# master_id                            0
# order_channel                        0
# last_order_channel                   0
# first_order_date                     0
# last_order_date                      0
# last_order_date_online               0
# last_order_date_offline              0
# order_num_total_ever_online          0
# order_num_total_ever_offline         0
# customer_value_total_ever_offline    0
# customer_value_total_ever_online     0
# interested_in_categories_12          0

df_.dtypes
#master_id                             object
#order_channel                         object
#last_order_channel                    object
#first_order_date                      object
#last_order_date                       object
#last_order_date_online                object
#last_order_date_offline               object
#order_num_total_ever_online          float64
#order_num_total_ever_offline         float64
#customer_value_total_ever_offline    float64
#customer_value_total_ever_online     float64
#interested_in_categories_12           object
#dtype: object

df_.info()
df_.value_counts()

#TASK 1.3: Omnichannel müşterilerinin her biri için toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df_["order_num_total_ever_omnichannel"] = df_["order_num_total_ever_online"] + df_["order_num_total_ever_offline"]
df_["customer_value_total_ever_omnichannel"] = df_["customer_value_total_ever_offline"] + df_["customer_value_total_ever_online"]

#TASK 1.4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
import datetime as dt
date_columns = df_.columns[df_.columns.str.contains("date")]
df_[date_columns] = df_[date_columns].apply(pd.to_datetime)

df_.head()

#TASK 1.5:Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.

cltv = df.groupby('order_channel').agg({'master_id': lambda x: x.nunique(),
                                        'order_num_total_ever_omnichannel': lambda x: x.sum(),
                                        'customer_value_total_ever_omnichannel': lambda x: x.sum()})
#TASK 1.6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.

df_.sort_values('customer_value_total_ever_omnichannel',ascending = False)[:10]


#TASK 1.7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.

df_.sort_values('order_num_total_ever_omnichannel',ascending = False)[:10]

#TASK 1.8: Veri ön hazırlık sürecini fonksiyonlaştırınız.

def data_preparation(dataframe):
 df_["order_num_total_ever_omnichannel"] = df_["order_num_total_ever_online"] + df_["order_num_total_ever_offline"]
 df_["customer_value_total_ever_omnichannel"] = df_["customer_value_total_ever_offline"] + df_["customer_value_total_ever_online"]
 date_columns = df_.columns[df_.columns.str.contains("date")]
 df_[date_columns] = df_[date_columns].apply(pd.to_datetime)

 return df

##########################################
## TASK 2:RFM Metriklerinin Hesaplanması #
##########################################

#TASK 2.1: Recency, Frequency ve Monetarytanımlarını yapınız.

#Recency: Kullanıcının son yaptığı alışveriş üzerinden haftalık bazında geçen zamandır.
#Frequency:
#Monetary:

#TASK 2.2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız.
##recencydeğerini hesaplamak için analiz tarihini maksimum tarihten 2 gün sonrası seçebilirsiniz.

df_["last_order_date"].max() # alışveriş yapılan son tarih
determined_date = dt.datetime(2021,6,1) # 2 gün sonrası
type (determined_date)
rfm = pd.DataFrame()
rfm["customer_id"] = df_["master_id"]
rfm["recency"] = (determined_date-df_["last_order_date"]).astype('timedelta64[D]')
rfm["frequency"] = df_["order_num_total_ever_omnichannel"]
rfm["monetary"] =df_["customer_value_total_ever_omnichannel"]

rfm.head()
rfm.shape
##########################################
## TASK 3:RF Skorunun Hesaplanması ######
##########################################

#TASK 3.1: Recency, Frequency ve Monetary metriklerini qcutyardımı ile 1-5 arasında skorlara çeviriniz.
#TASK 3.2: Bu skorları recency_score, frequency_scoreve monetary_scoreolarak kaydediniz.
#TASK 3.3: recency_scoreve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.describe().T

#########################################################
## TASK 4:  RF Skorunun SegmentOlarak Tanımlanması ######
#########################################################

#TASK 4.1: Oluşturulan RF skorları için segment tanımlamaları yapınız.
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

#TASK 4.2: seg_map yardımı ile skorları segmentlere çeviriniz

rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

rfm[rfm["segment"] == "new_customers"].head()

#########################################################
## TASK 5:  Aksiyon Zamanı ! ######
#########################################################

#TASK 1:Segmentlerin recency, frequency ve monetary ortalamalarını inceleyiniz.

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean","count"])

#TASK 2:RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun v emüşteri id'lerini csv olarak kaydediniz.

#CASE 1: FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği
# markanın ürün fiyatları genel müşteri tercihlerinin üstünde. Bu nedenle markanın
# tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
# iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve
# kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kurulacak müşteriler.
# Bu müşterilerin id numaralarını csv dosyasına kaydediniz.
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

requested_segment = rfm[rfm["segment"].isin(["loyal_customer","champions"])]["customer_id"]
last_requested = df_[(df_["master_id"].isin(requested_segment)) & (df_["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
last_requested.to_csv("new1_customers.csv", index = False)


#CASE 2:Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle
# ilgili kategorilerle ilgilenen geçmişte iyi müşteri olan ama uzun süredir alışveriş
# yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni gelen müşteriler
# özel olarak hedef alınmak isteniyor.Uygun profildeki müşterilerin id'lerini csv
# dosyasına kaydediniz.


requested_segment2 = rfm[rfm["segment"].isin(["about_to_sleep","new_customers","can't_loss_them"])]["customer_id"]
last_requested2 = df_[(df_["master_id"].isin(requested_segment2)) & ((df_["interested_in_categories_12"].str.contains("ERKEK"))| (df_["interested_in_categories_12"].str.contains("ÇOCUK")))]["master_id"]
last_requested2.to_csv("new2_customers.csv", index = False)


#function:

def rfm(dataframe):
    #veri hazırlama
    df_["order_num_total_ever_omnichannel"] = df_["order_num_total_ever_online"] + df_["order_num_total_ever_offline"]
    df_["customer_value_total_ever_omnichannel"] = df_["customer_value_total_ever_offline"] + df_[
        "customer_value_total_ever_online"]
    date_columns = df_.columns[df_.columns.str.contains("date")]
    df_[date_columns] = df_[date_columns].apply(pd.to_datetime)

    #rfm metriklerinin hesaplanması
    df_["last_order_date"].max()  # alışveriş yapılan son tarih
    determined_date = dt.datetime(2021, 6, 1)  # 2 gün sonrası
    type(determined_date)
    rfm = pd.DataFrame()
    rfm["customer_id"] = df_["master_id"]
    rfm["recency"] = (determined_date - df_["last_order_date"]).astype('timedelta64[D]')
    rfm["frequency"] = df_["order_num_total_ever_omnichannel"]
    rfm["monetary"] = df_["customer_value_total_ever_omnichannel"]

    #rfm skorunun hesaplanması

    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])

    rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    rfm["RF_SCORE"] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

    #segmentlerin belirlenmesi
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_Risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }
    #skorların segmentlere dönüştürülmesi

    rfm['segment'] = rfm['RF_SCORE'].replace(seg_map, regex=True)

    return rfm[["customer_id","recency","frequency","monetary","RF_SCORE","segment"]]






