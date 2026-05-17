import pandas as pd
import polars as pl
import datetime as dt

def load_data(dataset):
    """
    Fungsi untuk load dataset
    args:
        Parameter masukan berupa path raw dataset
    output:
        Dataframe pandas
    """
    # Load dataset 
    df = pl.read_csv(dataset, schema_overrides={
      'InvoiceNo': pl.String,
      'CustomerID': pl.String
    },
        encoding='utf8-lossy')

    # Ubah ke bentuk pandas
    df = df.to_pandas()

    return df

def preprocessing_data(dataset):
    """
    Fungsi untuk membersihkan raw dataset 
    Args:
        Parameter masukkan berupa raw dataframe
    Output:
        Dataframe yang sudah bebas dari outlier, nilai error, duplikat
    """
    # Hapus data duplikat
    dataset.drop_duplicates(inplace=True)
    # Hapus Null Values
    dataset.dropna(inplace=True)
    # Drop data non product / Error
    kode_non_produk = [
        'BANK CHARGES', # Tagihan Bank
        'POST',
        'D', # Discount
        'M', # Manual
        'DOT', # Dotcom postage
        'CRUK', # Cancer Research UK
        'PADS', # Pads to match Wrapping
        'C2', # Carriage
    ]
    dataset = dataset[~dataset['StockCode'].isin(kode_non_produk)]

    # Drop data None dalam 2 kolom ini
    dataset = dataset.dropna(subset=['Description', 'CustomerID'])
    # Hanya mengambil data yang tidak menghasilkan nilai (-)
    dataset = dataset[(dataset['UnitPrice'] >= 0.0) & (dataset['Quantity']>= 0)]

    # Hapus jam, menit, detik dari kolom
    dataset['InvoiceDate'] = pd.to_datetime(dataset['InvoiceDate'], format='%m/%d/%Y %H:%M')
    dataset['InvoiceDate'] = dataset['InvoiceDate'].dt.normalize()

    return dataset

def build_features_and_label(dataset, window_days=90):
    """
    Fungsi untuk membuat fitur RFM dan juga memberikan labeling churn
    Args:
        Parameter input berupa dataframe yang sudah bersih dari hasil preprocessing
        dan juga windows_days untuk menentukan pembagian waktu
    Output:
        Dataframe yang siap digunakan oleh model
    """
    # Cut Tanggal
    tanggal_max_data = dataset['InvoiceDate'].max()
    tanggal_cutoff = tanggal_max_data - dt.timedelta(days=window_days)

    data_observasi = dataset[dataset['InvoiceDate'] < tanggal_cutoff].copy()
    data_prediksi = dataset[dataset['InvoiceDate'] >= tanggal_cutoff].copy()

    data_observasi['Total_Harga'] = data_observasi['Quantity'] * data_observasi['UnitPrice']

    # Buat feature RFM
    rfm = data_observasi.groupby('CustomerID').agg(
        # Recency
        Tanggal_Pertama_Belanja=('InvoiceDate', 'min'),
        Tanggal_Terakhir_Belanja=('InvoiceDate', 'max'),

        # Frequensi
        Frequency=('InvoiceNo', 'nunique'),

        # Monetary
        Monetary=('Total_Harga', 'sum')
    ).reset_index()

    rfm['Recency'] = (tanggal_cutoff - rfm['Tanggal_Terakhir_Belanja']).dt.days
    rfm['Tenure'] = (tanggal_cutoff - rfm['Tanggal_Pertama_Belanja']).dt.days
    rfm['Average_Order_Value'] = rfm['Monetary'] / rfm['Frequency']

    # Daftar pelanggan
    pelanggan_kembali = data_prediksi['CustomerID'].unique()

    # Labelling Churn
    rfm['Churn'] = rfm['CustomerID'].apply(lambda x: 0 if x in pelanggan_kembali else 1)

    # Hapus fitur yang tidak digunakan
    rfm = rfm.drop(columns=['CustomerID', 'Tanggal_Pertama_Belanja', 'Tanggal_Terakhir_Belanja'])

    return rfm

if __name__ == '__main__':
    print("Memproses data")
    raw_data = load_data("../data.csv")
    clean_data = preprocessing_data(raw_data)
    data_final = build_features_and_label(clean_data, window_days=90)

    # Save data bersih
    data_final.to_csv("clean_data.csv", index=False)
    print("Selesai")
