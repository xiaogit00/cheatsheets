### Creating Arrays
`a = np.array([1,2,3]) `  
`b = np.array([(1.5,2,3), (4,5,6)], dtype = float)  `  
`c = np.array([[(1.5,2,3), (4,5,6)],[(3,2,1), (4,5,6)]], dtype = float) `   

### Initial Placeholders
`np.zeros((3,4))` - Create an array of zeros   
`np.ones((2,3,4),dtype=np.int16)` - Create an array of ones  
`d = np.arange(10,25,5)`- Create an array of evenly spaced values (step value)  
`np.linspace(0,2,9)` - Create an array of evenlyspaced values (number of samples)  
`e = np.full((2,2),7)`- Create a constant array  
`f = np.eye(2)` - Create a 2X2 identity matrix  
`np.random.random((2,2))` - Create an array with random values  
`np.empty((3,2))` - Create an empty array  
`np.random.randint(0, 10, 10)` - Create an array of 10 integers btwn 0 to 10  

### Reshaping
`np.arange(24).reshape(4, 2, 3)` - creates a 3D array, 4 blocks, each containing 3x2 elements

### Array slicing
`a[3, 2, 1]` - gets first block, second element, and 1st element 
`a[3, 0:1, 1]` - slices the second dimension

### Union/Intersection
`np.union1d(s2, s3)` - unique values in both s2 and s3 (union)  
`np.intersect1d(s2, s3)` - items common to both s2 and s3 (intersection)  

### Stats
`np.percentile(s1, q=[0, 25, 50, 75, 100])` - given s1, return an array of percentiles  