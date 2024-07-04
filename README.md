# Twitter Sentiment Trading Strategy

This project implements a trading strategy based on Twitter sentiment analysis. The strategy involves selecting stocks with high engagement ratios (calculated from Twitter comments and likes) and comparing their performance against the NASDAQ index (QQQ).

## Project Overview

The project reads sentiment data from a CSV file, processes the data to calculate engagement ratios, ranks the stocks based on these ratios, and then forms a portfolio consisting of the top-ranked stocks each month. The performance of this portfolio is then compared against the NASDAQ index.

## Files

- `sentiment_data.csv`: Contains the sentiment data including Twitter comments and likes for various stocks.
- `twitter_sentiment_trading.py`: The main script implementing the trading strategy.
- `README.md`: This file.

## Requirements

- Python 3.7+
- Pandas
- NumPy
- yfinance
- Matplotlib

Install the required packages using pip:

```sh
pip install pandas numpy yfinance matplotlib
```

## Data

The `sentiment_data.csv` file should have the following columns:

- `date`: Date of the sentiment data.
- `symbol`: Stock symbol.
- `twitterComments`: Number of comments on Twitter.
- `twitterLikes`: Number of likes on Twitter.

## Code Explanation

### Import Libraries

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt
import os
plt.style.use('ggplot')
```

### Read and Process Sentiment Data

```python
data_folder = 'C:\\Users\\Dell\\Downloads'
file_name = 'sentiment_data.csv'
sentiment_df = pd.read_csv(os.path.join(data_folder, file_name))
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df = sentiment_df.set_index(['date', 'symbol'])
sentiment_df['engagement_ratio'] = sentiment_df['twitterComments'] / sentiment_df['twitterLikes']
sentiment_df = sentiment_df[(sentiment_df['twitterLikes'] > 20) & (sentiment_df['twitterComments'] > 10)]
```

### Aggregate Data by Month

```python
aggragated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='ME'), 'symbol'])[['engagement_ratio']].mean())
aggragated_df['rank'] = aggragated_df.groupby(level=0)['engagement_ratio'].transform(lambda x: x.rank(ascending=False))
filtered_df = aggragated_df[aggragated_df['rank'] < 6].copy()
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index + pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])
```

### Prepare Dates and Symbols for Portfolio Construction

```python
dates = filtered_df.index.get_level_values('date').unique().tolist()
fixed_dates = {}
for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
stocks_list = sentiment_df.index.get_level_values('symbol').unique().tolist()
```

### Download Historical Price Data

```python
prices_df = yf.download(tickers=stocks_list, start='2021-01-01', end='2023-03-01')
returns_df = np.log(prices_df['Adj Close']).diff().iloc[1:]
```

### Construct Portfolio Returns

```python
portfolio_df = pd.DataFrame()
for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
    cols = fixed_dates[start_date]
    temp_df = returns_df[start_date:end_date][cols].mean(axis=1).to_frame('portfolio_return')
    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
```

### Compare Against NASDAQ Index (QQQ)

```python
qqq_df = yf.download(tickers='QQQ', start='2021-01-01', end='2023-03-01')
qqq_ret = np.log(qqq_df['Adj Close']).diff().to_frame('nasdaq_return')
portfolio_df = portfolio_df.dropna()
qqq_ret = qqq_ret.dropna()
portfolio_df = portfolio_df.merge(qqq_ret, left_index=True, right_index=True)
```

### Calculate and Plot Cumulative Returns

```python
cumulative_portfolio_return = (1 + portfolio_df['portfolio_return']).cumprod() - 1
cumulative_nasdaq_return = (1 + portfolio_df['nasdaq_return']).cumprod() - 1
cumulative_returns = pd.DataFrame({
    'Cumulative Portfolio Return': cumulative_portfolio_return,
    'Cumulative NASDAQ Return': cumulative_nasdaq_return
})
ax = cumulative_returns.plot(figsize=(16, 6), linestyle='-')
plt.title('Twitter Engagement Ratio Strategy Return Over Time')
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))
plt.ylabel('Return')
plt.xlabel('Date')
plt.legend(['Cumulative Portfolio Return', 'Cumulative NASDAQ Return'])
plt.grid(True)
plt.show()
```

## Running the Code

1. Ensure you have the required packages installed.
2. Place the `sentiment_data.csv` file in the specified folder.
3. Run the script `twitter_sentiment_trading.py`.

This will process the sentiment data, construct the trading strategy, and plot the cumulative returns compared to the NASDAQ index.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```

