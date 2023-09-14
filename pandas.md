### Series 
`s2 = pd.to_numeric(s1, errors='coerce')` - converting a series to a different type  

### Reading csv / excel
`pd.read_csv()`  
`pd.read_excel()`  
`pd.read_csv('sample.csv', usecols=['Time', 'Geo', 'Value'])`  
`df.set_index('X', inplace=True)` - sets X as index
`df.reset_index(inplace = True)`
`df.loc['Snow']` - locates by index, returns all 'Snow' index
`pd.read_csv(...nrows = 3)` - num of rows to read  
`pd.read_csv(... na_values=["not available", "na"])` - specify na values
`pd.read_csv(... na_values={'esp': ['na', 'not available'], 'rev': [-1]})` - specify type of na values in diff cols; expects a dict.  

### Saving
`dv.to_csv('new.csv')`  
`df.to_csv('new.csv', index=False)` - no index  
`df.to_csv('new.csv', columns=['tickers', 'eps'])` - export certain columns  
`df.to_excel('new.xlsx', sheet_name='stocks', startrow=1, startcol=2)`  
```
with pd.ExcelWriter('stocks_weather.xlsx') as writer:
    df.stocks.to_excel(writer, sheet_name='stocks')
    df.weather.to_excel(writer, sheet_name='weather')
```
Writing to different sheets in excel.   

### Creating Dataframes
`pd.DataFrame(dict)`  
`pd.DataFrame(dict, index = [])`
`pd.Series([])`
`pd.Series([], index = [], name='')`


### DF attributes
`df.columns` - an array of columns  
`df.index` - an array of its index  
`df.values` - an array of each row's values
`df.values[0]` - an array of first row's values

### Filtering / Selecting
`df['Values']` - selecting only *Values* column  
`df.Values` - selecting only Values column  
`df[['event', 'value']]` - selecting both columns  
`df[df.X == df.X.max()]` - get row where X is max
`df[10:14]` - selecing only row 10 to 13, by position (not index)  
`df.ix[90:94, ['Values', 'Time']]` - selecting rows 90:94 by index, and columns by index  
`df[df['Values'] > 5]` - returns only rows with *Values* more than 5  
`df[df['Values'].isnull()]` - returns only rows with *Values* is null  
`df['col'][0]` - row 0 of column col1  
`df.iloc[0]` - select row  
`df.iloc[:, 0]` - left of comma: row, right of comma: column. Select everything of col 1  
`df.iloc[1:3, 0]` - row 1, 2 of col0  
`df.iloc[[0, 1, 2], 0]` - rows 0, 1, 2 of col0  
`df.loc[0, 'country]` - value at desired location: here, row 0, country.  

### Quick stats / Aggregation
`df.describe()` - quick stats  
`df.count()` - num of non-null observations  
`df.sum()`  
`df.mean()`  
`df.median()`  
`df.min()`  
`df.max()`  
`df.prod()` - product of values  
`df.std()` - unbiased standard deviation   
`df.var()` - unbiased variance  
`df.max(axis=0)` - max for each column  
`df.max(axis=1)` - max for each row  
`df.item_name.nunique()` - count of unique in series  

### Arithmatic
`s = edu['value']/100` - applies operation over value column, and return a series  
`s = edu['value'].apply(np.sqrt)` - applies sqrt function using .apply() on column
`s = edu['value'].apply(lambda d: d**2)` - applying a lambda function to 'value' column  


### Adding/Removing columns
`edu['ValueNorm'] = edu['Value']/edu['Value'].max()` - dividing the value column by max value of same column to get a series, assigning it to new column  
`edu.drop('ValueNorm', axis = 1)` - axis 0 removes rows, axis 1 removes columns. Return a copy of the modified data. 
`edu.drop('ValueNorm', axis = 1, inplace = True)` - does not return copy of data. 

### Appending/removing rows
`edu = edu.append({"Time": 2000, "Value": 5.00}, ignore_index=True)` - expects a dict where the keys are the name of the columns, and the values being values. If *ignore_index* is not specified, index 0 will be given to this new row, and will produce an error if it already exists.  
`edu.drop(max(edu.index), axis = 0, inplace=True)` - remove last row  
`eduDrop = edu.drop(edu['Value'].isnull(), axis = 0)` - copy of df without NaN values.  

### Dropping all na values
`edu.dropna(how = 'any', subset = ['Value'])` - erase any row that contains an NaN value, for the "Value" column  

### Dropping duplicates
`df.drop_duplicates(subset=['col1'])` - drop duplicates of col1  

### Fillna 
`edu.fillna(value = {"Value": 0})` - value takes a dict, with column being the name of the column to fill, and the value to fill.  
Using Regex:  
```
df.replace({
    'temperature': '[A-Za-z]',
    'windspeed': '[A-Za-z]',
}, '1000', regex = True)
```  
`df.replace(['poor', 'averge', 'good', 'exceptional'], [1,2,3,4])` - searches entire df with these values and replaces with latter  

### Replacing certain values by sth else
```
def convert_people_cell(cell):
    if cell == 'n.a.':
        return "Sam"
    return cell 

df = pd.read_csv('sample.csv', converters = {
    'people': convert_people_cell
})
```

### Sorting
`edu.sort_values(by='Value', ascending = False, inplace = True)`  
`edu.sort_index(axis = 0, ascending = True, inplace = True)` - return to original order (sort by index using sort_index and axis = 0)

### Grouping data
`edu[["Geo", "Value"]].groupby("Geo").mean()` - must always have an aggregation function applied.  
`g.get_group('mumbai)` - gets only this group  

### Pivoting
`pivedu = pd.pivot_table(edu[edu['Time']>2005], values = "Value", index = ['Geo'], columns = ['Time'])` - changing a previous column values (GEO) to index, and a column value (time) to column. The values will be from the value column.  

Pivot has `aggr_function` argument that allows us to perform aggregation function between values if there's more than 1 value for the given row/column after the transformation.  

`df.pivot(index = 'date', columns = 'city')`


### Ranking
`edu.rank(ascending = False, method = 'first')`  

### Concat & Merge
`df3 = pd.concat([df1, df2])`  
`df3 = pd.concat([df1, df2], ignore_index = True)`  
`df3 = pd.concat([df1, df2], keys = ["india", "us"])`  
`pd.concat([df, s], axis = 1)` - Adding a series to DF  
`df3 = pd.merge(df1, df2, on='city')`    
`df5 = pd.merge(df1, df2, on='city', how='left')`    


### Plotting
`totalSum.plot(kind='bar', style='b', alpha=0.4, title = 'Total values for Country)` - style = color of bars set to blue; alpha is a percentage, producing a more translucent plot  

### Periods
- a class that represents a duration of time  
`y = pd.Period(2016)` - a whole year  
`pd.Period(2016).start_time` - returns *2016-01-01 00:00:00*  
`pd.Period(2016).end_time` - returns *2016-12-31 23:59:59*  
`m = pd.period('2011-1', freq='M')` - a month
`m + 1` - returns *Period('2011-02', 'M')*
`pd.period('2011-1-01', freq='D')`
`q = pd.Period('2017Q1')`  
`d.asfreq('M', how='start')` - convert to month frequency  

### Period index
`idx = pd.period_range('2011', '2017', freq='Q-Jan)` - create index of each quarter from 2011 to 2017  
`pd.period_range('2011', '2017', freq='M')`  

### Datetime
`df.X = pd.to_datetime(df.X)` - converts date to datetime  
`df.date = pd.DatetimeIndex(df.Date)` - sets index as datetime  
`price['{}'.format(startYr):'{}'.format(endYr)]` - using var to filter  
`price['1980']` - returns records for that year  
`price['1980-01']` - all prices for the month
`price['1980':'2000']` - rance of prices  
`datetime.datetime(2001,1,1)` - converting to datetime  
`'2001-01-01`.date() -> convert string to date  
`date.year` - extracts year from date  

### Resampling
`s.price.resample('BY').first().ffill()` - returns yearly price, takes first value  
Resampling frequency:
`B` - business day freq  

### Valid business days
`import pandas_market_calendars as mcal`  
`nyse = mcal.get_calendar('NYSE')`  
`startDate = nyse.valid_days(start_date='2001-01-01', end_date='2001-12-01)`  
