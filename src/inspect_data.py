from src.data.load import load_sample


def main():
    print("Loading dataset sample...")

    df = load_sample(10)

    print("\nDataset sample shape:")
    print(df.shape)

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}")

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nMissing values:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()