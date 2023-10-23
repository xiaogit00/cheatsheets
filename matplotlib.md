## Basics
pyplot is an API for matplotlib that provides simple functions for adding plot elements, such as lines, images, etc, to the axes of the current figure. It is defined by statefulness, meaning it stores the state of the object when you first plot it. This allows you to continuously plot over the same chart. 

### Key Concepts
**Figures**
The top level object in the scripting layer is the pyplot.figure(). The Figure is an object that keeps the whole image output. 

**Axes**
An Axes object represents the pair of axis that contain the single plot (x-axis and y-axis). This represents an individual plot (not to be confused with x and y axis). We call methods that do the plotting directly from the Axes. 
It has methods like: 
- `set_xlim()` - x and y axis limits
- `set_xlabel()` - x and y axis labels
- `set_title()` - plot title

In sum, a figure is a canvas, and an Axes is a part of the canvas on which we make a particular visualization. 

``` python
# Example of using the Axes
fig, ax = plt.subplots() #Creates an empty canvas, and an axes instance.
ax.barh(name, data)

#OR
fig = plt.figure(figsize =(5, 4))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # this array input is the dimensions (left bottom, width, height) of the new axes)
ax.plot(x, y)
ax.set_xlim(1, 2)
ax.set_xticklabels(labels)

```


The df.plot() method in Pandas Series and DataFrame are just a simple wrapper around plt.plot(). 

**Examples**
`plt.gcf()` - gets you a reference to the current figure when using pyplot
``` python
plt.scatter(x, y) #this creates a figure in the background

fig = plt.gcf() # retrieve the figure created by the call above
fig.set_size_inches(6, 2) # set certain things on it
```

`plt.gca()` - gets you a reference to the current axes, if you need to change the limits on the y-axis, for example
``` python
plt.scatter(x, y)
axis = plt.gca()
axis.set_ylim(-3,3)
plt.show()
```
`plt.cla()` and `plt.clf()`  
- clear current axes or clear current figure
- These methods are used to clear the current figure (plt.clf()) or the current axes (plt.cla()) so the previous plots don't interfere with the next ones. 

## Importing
`import matplotlib.pyplot as plt` - import pyplot  
`import matplotlib as mpl`  

## Plotting
`plt.plot([1,2,3])` - assumes to be x values     
`plt.plot(x, y)` - takes an array of x and y values    

## Subplots  
`plt.subplot(nrow, ncol, index)`
- for adding more than 1 figure in a plot. Inputs: the num of rows and cols you want in your subplot.    

## Changing Title & Labeling
`plt.title("My plot")`  
`plt.ylabel('Y Axis')`  
`plt.xlabel('X Axis')`  
`labels = ax.get_xticklabels()`  - get the current labels, for further styling for instance  
`plt.setp(labels, rotation=45, horizontalalignment='right')` - tilts the x axis labels to the right  

## Figure size
`plt.figure(figsize=())`

## Showing diagram
`plt.show()`

## Styling
#### rcParams
Styling info is contained in the matplotlib dictionary-like runtime configuration variable: `matplotlib.rcParams`. Here is the [reference](https://matplotlib.org/stable/api/matplotlib_configuration_api.html#matplotlib.rcParams). 

You can set it as follows:
``` python
import matplotlib as mpl

mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.linestyle'] = '--'
```

You can also change multiple values at once using the `mpl.rc` function. It uses keyword arguments.  
`mpl.rc('lines', linewidth=4, linestyle='-.)`

#### Style sheets
The other way is to edit the style sheet.
`plt.style.use('ggplot')`

You can create your own style and call style.use:
For instance, you can create a `custom.mplstyle` file that takes the following:
```
axes.titlesize : 24
axes.labelsize : 20
lines.linewidth : 3
lines.markersize : 10
xtick.labelsize : 16
ytick.labelsize : 16
```
And import it into your work like so:  
`plt.style.use('./custom.mplstyle')`

## Pandas DFs
df.plot() - normal line plot


## Common scripts
``` python
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import style

```