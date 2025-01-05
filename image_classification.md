# Three ways to do Image classification

## Frontmatter: creating your datasets

``` python
from fastai.vision.all import *

# Importing your images and converting them into p*n*m tensors 
three_tensors = [tensor(Image.open(o)) for o in Path('/Users/lei/.fastai/data/mnist_sample/train/3').ls().sorted()] # [#6131 x (28*28)]
stacked_threes = torch.stack(three_tensors).float()/255 # Shape: 6131*28*28
seven_tensors = [tensor(Image.open(o)) for o in Path('/Users/lei/.fastai/data/mnist_sample/train/7').ls().sorted()] # [#6131 x (28*28)]
stacked_sevens = torch.stack(seven_tensors).float()/255 # Shape: 6265*28*28

# Creating your training set
train_x = torch.cat([stacked_threes, stacked_sevens]).view(-1, 28*28) # Shape: 12396 * 784
train_y = tensor([1]*len(stacked_threes) + [0]*len(stacked_sevens)).unsqueeze(1) # Converts a rank 1 tensor into a rank 2 tensor; [12396] -> (12396, 1)

# Creating your data sets
dset = list(zip(train_x, train_y)) # An array of tuples 

# Creating your validation sets
valid_three_tensors = [tensor(Image.open(o)) for o in Path('/Users/lei/.fastai/data/mnist_sample/valid/3').ls().sorted()] # [#6131 x (28*28)]
valid_seven_tensors = [tensor(Image.open(o)) for o in Path('/Users/lei/.fastai/data/mnist_sample/valid/7').ls().sorted()] # [#6131 x (28*28)]
stacked_valid_threes = torch.stack(valid_three_tensors).float()/255
stacked_valid_sevens = torch.stack(valid_seven_tensors).float()/255
valid_x = torch.cat([stacked_valid_threes, stacked_valid_sevens]).view(-1, 28*28)
valid_y = tensor([1]*len(valid_three_tensors) + [0]*len(valid_seven_tensors)).unsqueeze(1)
valid_dset = list(zip(valid_x, valid_y))
```

## 1. Low Level Approach

```python

# Creating your weights and bias 
def init_params(size, std=1.0):
    return (torch.randn(size)*std).requires_grad_()

weights = init_params((28*28, 1)) # Shape: 784*1
bias = init_params(1) # Shape: [1]

# Defining your linear model (y_hat)
def linear1(xb):
    return xb@weights + bias


# Defining your loss function
def mnist_loss(predictions, targets):
    predictions = sigmoid(predictions)
    return torch.where(targets==1, 1-predictions, predictions).mean() # this returns a scalar, in this case, accuracy figure i.e. 0.45
    # 'where' is simply a function to check for whether targets==1; if it is, it's b, else it's c
    # It's essentially the same as your accuracy!

# Creating the mini-batching infrastructure for better training
dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

# Creating the training lifecycle 
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward() # The way to interpret this is, run dΘ/dy_hat for all params
        # you'll have access to the params: i.e. weights, and bias, which are the objects that are mutated
        # Recap: weights shape: (784, 1)

def train_epoch(model, lr, params):
    for xb, yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr #update gradient
            p.grad.zero_() # Make grad 0, if not, the loss.backward adds the gradient to the previously stored grad

# Creating a batch accuracy function
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds>0.5) == yb
    return correct.float().mean() # Basically the accuracy function - but same as mnist_loss? 

# Creating a validate epoch engine
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

# Training and validation
validate_epoch(linear1)

lr = 1.
params = weights,bias
for i in range(20):
    train_epoch(linear1, lr, params)
    print(validate_epoch(linear1), end=' ')
```

## 2. Pytorch

```python
# Defining your linear model (y_hat)
linear_model = nn.Linear(28*28, 1) # this creates your weights and biases

# Defining your loss function
def mnist_loss(predictions, targets):
    predictions = sigmoid(predictions)
    return torch.where(targets==1, 1-predictions, predictions).mean() 

dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

# Creating the training lifecycle 
def calc_grad(xb, yb, model):
    preds = model(xb)
    loss = mnist_loss(preds, yb)
    loss.backward()


# Creating a batch accuracy function
def batch_accuracy(xb, yb):
    preds = sigmoid(xb)
    correct = (preds>0.5) == yb
    return correct.float().mean() # Basically the accuracy function - but same as mnist_loss? 

# Creating a validate epoch engine
def validate_epoch(model):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)

class BasicOptim:
    def __init__(self, params, lr):
        self.params, self.lr = list(params), lr
    
    def step(self, *args, **kwargs):
        for p in self.params: 
            p.data -= p.grad.data * self.lr
    
    def zero_grad(self, *args, **kwargs):
        for p in self.params:
            p.grad = None

lr = 1.0

opt = BasicOptim(linear_model.parameters(), lr)

def train_epoch(model):
    for xb, yb in dl:
        calc_grad(xb, yb,model)
        opt.step()
        opt.zero_grad()

def train_model(model, epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model), end=' ')

train_model(linear_model, 20)
```

## 3. fastai

```python
# Defining your loss function
def mnist_loss(predictions, targets):
    predictions = sigmoid(predictions)
    return torch.where(targets==1, 1-predictions, predictions).mean()

# Creating the mini-batching infrastructure for better training
dl = DataLoader(dset, batch_size=256)
valid_dl = DataLoader(valid_dset, batch_size=256)

# Creating a batch accuracy function
def batch_accuracy(xb, yb):
    preds = sigmoid(xb)
    correct = (preds>0.5) == yb
    return correct.float().mean() # Basically the accuracy function - but same as mnist_loss? 

dls = DataLoaders(dl, valid_dl)

learn = Learner(dls, nn.Linear(28*28, 1), opt_func=SGD, loss_func=mnist_loss, metrics=batch_accuracy)

learn.fit(10, lr=1.0)
```
