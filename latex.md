### Entering latex environment in Jupyter Notebook:
```
$$
Enter Latex here
$$
```
Inline latex:  
$Enter latext formula here$


### Fractions:
`PV = \frac{FV}{(1+r)^n}`

### Fractions with exponent expressions:
`\frac{1}{(1+r)^{k-1}}`  

### Equation Numbering
Add the following to a cell:  
```
%%javascript
MathJax.Hub.Config({
    TeX: { equationNumbers: { autoNumber: "AMS" } }
})
```
Then you can call it like so:
```\begin{equation}
PV = \frac{FV}{(1+r)^n}
\tag{2}
\end{equation}
```

### Escaping the dollar sign in Jupyter Notebook
//$

