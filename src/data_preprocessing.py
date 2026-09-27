import pandas as pd


def load_data():
    df = pd.read_csv("data/telco_churn.csv", sep=";")

    print("Kích thước dữ liệu:", df.shape)
    print("\n5 dòng đầu tiên:")
    print(df.head())

    print("\nThông tin dữ liệu:")
    print(df.info())

    print("\nSố lượng khách hàng theo nhãn Churn:")
    print(df["Churn Label"].value_counts())

    return df


if __name__ == "__main__":
    load_data()