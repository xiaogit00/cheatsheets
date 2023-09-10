## Os Module
Helps with directory, path, folder management. 

### Importing
import os

### Creating paths that's cross compatible between Mac and Windows
`os.path.join('usr', 'bin', 'spam')`

Windows: `usr\\bin\\spam`
Mac: `usr/bin/spam`

### Get current directory
`os.getcwd()`

### Change directory
`os.chdir('/Users/lei/Downloads)`

### Creating folders
`os.makedirs('/photos/december/bali')`  
This will create all the following nested folders. 

### Absolute & Relative paths
`os.path.abspath(path)` - returns absolute path of argument. Easiest way to convert rel path to absolute.  
`os.path.isabs(path)` - return true if path is absolute  
`os.path.relpath(path, start)` - returns string of rel path from 'start' to 'path'.   

### Dir & base names
`os.path.basename(path)` - file name  
`os.path.dirname(path)` - dir name  
`os.path.split(filePath)` - returns dir name and file name as tuple   
`filePath.split(os.sep)` - returns an array of (dir) folders & file name   

### File size:
`os.path.getsize(path)` - size in bytes of the file in path
`os.listdir(path)` - return a list of filename strings 

### Path validity:
`os.path.exists(path)` - checks if file/folder exists
`os.path.isfile(path)` - return true if path is file
`os.path.isdir(path)` - return true if path arg exists and is a folder 

### File Reading
`helloFile = open('file.txt')`
`content = helloFile.read()`
`helloFile.readlines()`

### File Writing
`sampleFile = open('sample.txt', 'w')`
`sampleFile.write('Hello')`
`sampleFile.close()`

### Shelve module
`import shelve`
`shelfFile = shelve.open('mydata')` - creates a shelf file *mydata.db* at pwd
`shelfFile['cats'] = ['cat1', 'cat2', 'cat3']` - saves the value into 'cats' key
`shelfFile.close()`

To consume:
`import shelve`
`shelfFile = shelve.open('mydata)`
`shelfFile['cats']`
