import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preprocessing(df1,df2,df3) -> pd.DataFrame:

    #loading dataset
    df1 = pd.read_csv('C:/projects/ml-pipeline/data/raw/nykaa_campaign_data_with_nulls.csv', encoding = 'utf-8')
    df2 = pd.read_csv('C:/projects/ml-pipeline/data/raw/purplle_campaign_data_with_nulls.csv', encoding= 'utf-8')
    df3 = pd.read_csv('C:/projects/ml-pipeline/data/raw/tira_campaign_data_with_nulls.csv', encoding= 'utf-8')

    df1["Brand"] = "Nykaa"
    df2["Brand"] = "Purplle"
    df3["Brand"] = "Tira"

    #combine dataset
    df= pd.concat([df1,df2,df3], ignore_index= True)

    #drop duplicates
    df= df.drop_duplicates()

    #dropping unwanted columns
    df= df.drop('Campaign_ID', axis = 1)

    #splitting columns
    num_cols= df.select_dtypes(exclude=['object', 'datetime64[ns]']).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    date_cols = df.select_dtypes(include=['datetime64[ns]'])

    for col in num_cols:   #using median for fill missing values in numerical columns
      df[col]= df[col].fillna(df[col].median())  

    for col in cat_cols:    #using proportionate method to fill missing values in categorical columns
      proportions= df[col].dropna().value_counts(normalize=True)
      missing_mask = df[col].isnull()
      df.loc[missing_mask, col] = np.random.choice(
                 proportions.index, size = missing_mask.sum(), p= proportions.values)

    date_cols = df.select_dtypes(include=['datetime64[ns]'])  
    df['Date'] = df['Date'].ffill()  #using forward fill to handle date column


    #fixing categorical columns
    for col in cat_cols:
      df[col] = df[col].str.lower()
      df[col] = df[col].str.strip()

    #fixing Channel_Used column
    df['Channel_Used'] = df['Channel_Used'].fillna('Unknown')
    df['Channel_Used'] = df['Channel_Used'].str.split(",")
    df["Channel_Used"] = df["Channel_Used"].apply(lambda x: sorted([i.strip() for i in x]))
    df["Channel_Used"] = df["Channel_Used"].apply(lambda x: ", ".join(x))
    
    #splitting based on number of channels used
    multiple = df["Channel_Used"].apply(lambda x: len(x.split(",")) > 1).sum()
    single   = df["Channel_Used"].apply(lambda x: len(x.split(",")) == 1).sum()

    print(f"Single channel rows: {single}")
    print(f"Multiple channel rows: {multiple}")

    #saving cleaned dataframe
    df.to_csv("data/processed/clean_df.csv", index= False)
    print("✅ Cleaned data saved!")

    return df  


if __name__ == "__main__":
   

   #loading dataset
    df1 = pd.read_csv('C:/projects/ml-pipeline/data/raw/nykaa_campaign_data_with_nulls.csv', encoding = 'utf-8')
    df2 = pd.read_csv('C:/projects/ml-pipeline/data/raw/purplle_campaign_data_with_nulls.csv', encoding= 'utf-8')
    df3 = pd.read_csv('C:/projects/ml-pipeline/data/raw/tira_campaign_data_with_nulls.csv', encoding= 'utf-8')

    #calling preprocessing function
    clean_df = preprocessing(df1,df2,df3)

    print(f"✅ Clean data saved! Shape: {clean_df.shape}")
   








