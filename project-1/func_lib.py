import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pandas.tseries.offsets import DateOffset
from pathlib import Path
from typing import List


def load_historical_data(filepath: str) -> pd.DataFrame:
    '''
        Loads historical data.

        Inputs:
            filepath: filepath to the csv data
            
        Returns:
            df: dataframe to be operated on
        
        
        Used for tasks:
        - Load historical GDP data
        - Load the historical stock price data for Apple and Microsoft
    '''
    df = pd.read_csv(filepath)
    df.title = Path(filepath).stem.upper()
    df.columns = df.columns.str.upper()
    return df

    
def explore_data(df: pd.DataFrame):
    '''
        Explores dataframe.
        
        Inputs:
            df: dataframe to be operated on
            
        Returns:
            none
        
        Used for tasks:
        - Use methods like .info() and .describe() to explore the data
    '''
    
    print(df.title) 
    print("\n--- DataFrame Info ---")
    df.info()
    print("\n--- Descriptive Statistics ---")
    print(df.describe(include='all'))  # includes non-numeric columns too


def check_for_nulls(df: pd.DataFrame):
    '''
        Checks for nulls in dataframe.
        
        Inputs:
            df: dataframe to be operated on
            
        Returns:
            none
        
        Used for tasks:
        - Check for nulls
    '''
    print(df.title)
    if df.isnull().values.any():
        print(df.isnull().sum())
    else:
        print('OK - No nulls')


def forward_fill(df: pd.DataFrame):
    '''
        Forward fills missing data.
        
        Inputs:
            df: dataframe to be operated on
            
        Returns:
            none - changes to the dataframe are done in place
         
        Used for tasks:
        - Forward fill any missing data
    '''
    df.ffill(inplace=True)


def convert_dollar_columns_to_numeric(df: pd.DataFrame, numeric_columns: List[str]):
    '''
        Removes dollar signs ('$') from a list of columns in a given dataframe AND casts the columns to a numeric datatype.
        Updates dataframe IN PLACE.
        
        Inputs:
            df: dataframe to be operated on
            numeric_columns: columns that should have numeric data but have dollar signs currently
            
        Returns:
            none - changes to the dataframe are done in place

        Used for tasks:
        - Remove special characters and convert to numeric/datetime 
    '''

    for numeric_column in numeric_columns:
       df[numeric_column] = df[numeric_column].replace({'\$': ''}, regex=True).astype(float)


def convert_datetime_columns_to_datetime(df: pd.DataFrame, datetime_columns: List[str]):
    '''
        Casts the columns to a datetime datatype.
        Updates dataframe IN PLACE.
        
        Inputs:
            df: dataframe to be operated on
            datetime_columns: columns that should have datetime data but are defined as object
            
        Returns:
            none - changes to the dataframe are done in place

        Used for tasks:
        - Use pandas's to_datetime() to convert any columns that are in a datetime format
    '''

    for col in datetime_columns:
        df[col] = pd.to_datetime(df[col])


def align_date_to_end_of_month(df: pd.DataFrame, cols: List[str]):
    '''
        Aligns date to end of the month.
        
        Inputs:
            df: dataframe to be operated on
            cols: columns that should be offset to end of the month
            
        Returns:
            none - changes to the dataframe are done in place
        
        Used for tasks:
        - Align inflation data so that the date is the month end (e.g. Jan 31, Feb 28/28)
    '''
    for col in cols:
        df[col] = df[col] + pd.offsets.MonthEnd(0)
        

def upsample_monthly_to_weekly(df: pd.DataFrame, col: str) -> pd.DataFrame:
    '''
        Upsamples from monthly to weekly.
        
        Inputs:
            df: dataframe to be used as input
            col: column that should be set as index in a new dataframe
            
        Returns:
            df: a copy of dataframe with weakly datapoints

        Used for tasks:
        - Upsample and interpolate from monthly to weekly 
    '''
    df_copy = df.copy()
    df_copy.set_index(col, inplace=True)
    df_copy = df_copy.resample('W').asfreq()
    df_copy = df_copy.interpolate(method='linear')
    return df_copy


def downsample_monthly_to_quarterly(df: pd.DataFrame, col: str) -> pd.DataFrame:
    '''
        Downsamples from monthly to quarterly.
        
        Inputs:
            df: dataframe to be used as input
            col: column that should be set as index in a new dataframe
            
        Returns:
            df: a copy of dataframe with quarterly datapoints
        
        Used for tasks:
        - Downsample from monthly to quarterly 
    '''
    df_copy = df.copy()
    df_copy.set_index(col, inplace=True)
    return df_copy.resample('QE').mean()
    

def get_last_x_months_of_data_by_column(df: pd.DataFrame, date_col:str, last_x_months: int) -> pd.DataFrame:
    '''
        Gets last X months of data.
        
        Inputs:
            df: dataframe to be used as input
            date_col: date column that will be used for calculation
            
        Returns:
            df: a copy of dataframe only x months of data
        
        Used for tasks:
        - Use the max date calculated above to get the last three months of data in the dataset 
    '''
    
    max_date = df[date_col].max()
    start_date = max_date - DateOffset(months=last_x_months)

    df_copy = df.copy()

    return df_copy[df_copy[date_col] > start_date]


def get_last_x_months_of_data(df: pd.DataFrame, last_x_months: int) -> pd.DataFrame:
    '''
        Gets last X months of data based on the index (assumed to be a datetime index).

        Inputs:
            df: dataframe to be used as input
            last_x_months: number of months to include from the latest date in the index

        Returns:
            df: a copy of dataframe containing only the last X months of data

        Used for tasks:
        - Use the max date calculated above to get the last three months of data in the dataset 
    '''
    max_date = df.index.max()
    start_date = max_date - DateOffset(months=last_x_months)

    return df[df.index > start_date].copy()


def plot_line_open_vs_close(df: pd.DataFrame, stock: str):
    '''
        Plots a line plot for open vs close.
        
        Inputs:
            df: dataframe to be used as input
            stock: name of a stock
            
        Returns:
            none 
        
        Used for tasks:
        - Plot time series of open v. close stock price for Apple using the last 3 months of data
    '''
    plt.figure(figsize=(12, 6))
    plt.plot(df['DATE'], df['OPEN'], label='Open', linewidth=2, color='blue')
    plt.plot(df['DATE'], df['CLOSE/LAST'], label='Close', linewidth=2, color='red')
    plt.title(f'Open vs. Closed Over Time for {stock}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
def plot_histogram_close(df: pd.DataFrame, stock: str):
    '''
        Plots a histagram for for open vs close.
        
        Inputs:
            df: dataframe to be used as input
            stock: name of a stock
            
        Returns:
            none 
        
        Used for tasks:
        - Plot time series of open v. close stock price for Apple using the last 3 months of data
    '''
    plt.figure(figsize=(10, 6))
    plt.hist(df['CLOSE/LAST'], bins=50, edgecolor='blue')
    plt.title(f'Histogram of Closing Prices for {stock}')
    plt.xlabel('Closing Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def calculate_percent_change(df: pd.DataFrame, period: int = 1): 
    '''
        Calculates percent change.
        
        Inputs:
            df: dataframe to be operated on 
            period: period to be used to calculate percent change 
            
        Returns:
            none - changes to the dataframe are done in place
        
        Used for tasks:
        - Calculate daily returns for Apple and Microsoft and the percent change in inflation from month to month
    '''
    df[f'{str(period)}_DAY_RETURN'] = df['CLOSE/LAST'].pct_change(period)


def resample_returns_daily_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Resamples daily returns to monthly returns.

        Inputs:
            df: daily returns dataframe with '1_DAY_RETURN' and datetime index

        Returns:
            DataFrame with datetime index at monthly frequency and '1_MONTH_RETURN' column
    '''
    monthly_returns = (1 + df['1_DAY_RETURN']).resample('ME').prod() - 1
    return monthly_returns.to_frame(name='1_MONTH_RETURN')

    
def set_index_and_sort_by_index(df: pd.DataFrame, col):
    '''
        Sets index on the dataframe and sorts by index.
        
        Inputs:
            df: dataframe to be operated on
            col: column to be used as index
            
        Returns:
            none - changes to the dataframe are done in place
        
        Used for tasks:
        - Interpolate stock returns from daily to monthly
    '''
    df.set_index(col, inplace=True)
    df = df.sort_index(inplace=True)

    
def plot_heatmap(df_corr: pd.DataFrame):
    '''
        Plots a histagram for for open vs close.
        
        Inputs:
            df: dataframe containing correlation matrix
            
        Returns:
            none 
        
        Used for tasks:
        - Plot heatmap
    '''
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

    
def calculate_rolling_volatility(df: pd.DataFrame, window = 5):
    '''
        Calculates rolling volatility.
        
        Inputs:
            df: dataframe to be operated on
            window: window to be used for calculating rolling volatility 
            
        Returns:
            none - changes to the dataframe are done in place
        
        Used for tasks:
        - Calculate rolling one-week volatility
    '''
    df[f'{window}_DAY_ROLLING_VOLATILITY'] = df['1_DAY_RETURN'].rolling(window=5).std()

def plot_rolling_volatility(df: pd.DataFrame):
    '''
        Plots a line plot of rolling weekly volatility against closing price.
        
        Inputs:
            df: dataframe used as input
            
        Returns:
            none 
        
        Used for tasks:
        - Plot the calculated rolling weekly volatility of Apple's closing price against Apple's closing price
        - Plot these on the same chart, but using different y-axes
    '''
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot closing price on left y-axis
    ax1.plot(df.index, df['CLOSE/LAST'], color='blue', label='Close Price')
    ax1.set_ylabel('Close Price', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Create second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    
    # Plot rolling volatility on right y-axis
    ax2.plot(df.index, df['5_DAY_ROLLING_VOLATILITY'], color='red', label='5-Day Volatility')
    ax2.set_ylabel('5-Day Rolling Volatility', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Title and layout
    plt.title('Close Price vs 5-Day Rolling Volatility')
    fig.tight_layout()
    plt.grid(True)
    plt.show()

    
def export_data(df: pd.DataFrame, filename:str):
    '''
        Exports dataframe as csv.
        
        Inputs:
            df: dataframe used as input
            filename: name of file to be used
            
        Returns:
            none 
        
        Used for tasks:
        - Export data
    '''
    df.to_csv(filename, encoding='utf-8-sig')