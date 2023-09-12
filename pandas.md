### Reading csv / excel
`pd.read_csv()`  
`pd.read_excel()`  
`pd.read_csv('sample.csv', usecols=['Time', 'Geo', 'Value'])`  

### DF attributes
`df.columns` - an array of columns  
`df.index` - an array of its index  
`df.values` - an array of each row's values
`df.values[0]` - an array of first row's values

### Filtering
`df['Values']` - selecting only *Values* column  
`df[10:14]` - selecing only row 10 to 13, by position (not index)
`df.ix[90:94, ['Values', 'Time']]` - selecting rows 90:94 by index, and columns by index  
`df[df['Values'] > 5]` - returns only rows with *Values* more than 5  
`df[df['Values'].isnull()]` - returns only rows with *Values* is null  

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

### Fillna 
`edu.fillna(value = {"Value": 0})` - value takes a dict, with column being the name of the column to fill, and the value to fill.  

### Sorting
`edu.sort_values(by='Value', ascending = False, inplace = True)`  
`edu.sort_index(axis = 0, ascending = True, inplace = True)` - return to original order (sort by index using sort_index and axis = 0)

### Grouping data
`edu[["Geo", "Value"]].groupby("Geo").mean()` - must always have an aggregation function applied.  

### Rearranging data (Pivoting)
`pivedu = pd.pivot_table(edu[edu['Time']>2005], values = "Value", index = ['Geo'], columns = ['Time'])` - changing a previous column values (GEO) to index, and a column value (time) to column. The values will be from the value column.  

Pivot has `aggr_function` argument that allows us to perform aggregation function between values if there's more than 1 value for the given row/column after the transformation.  

### Ranking
`edu.rank(ascending = False, method = 'first')`  

### Plotting
`totalSum.plot(kind='bar', style='b', alpha=0.4, title = 'Total values for Country)` - style = color of bars set to blue; alpha is a percentage, producing a more translucent plot