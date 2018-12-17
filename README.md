# Stanford-CS230-Project
**It's a bird... It's a plane... It's a Type Ia supernova!**

Astronomical Object Classification from Photometric Time-Series Data

## Instructions
Run the IPython notebook Models.ipynb cell-by-cell. The notebook will load the data from csv, perform the necessary transformations, and define necessary helper functions. Then, PyTorch NN modules are defined and trained. Along the way, the notebook saves model files, and outputs the model evaluation results. 

You can extend the code with your own PyTorch model, then simply call 
```python
model = YourNewModel(<params>)
train_model(model)
```

Note that the `forward` method of your model should expect to be passed a tuple of (data tensor, metadata tensor). The data tensor has dimensions (batch_size, n_channels = 6, T = 352), and the metadata tensor has dimensions (batch_size, n_metadata_features = 16).

Here is a simple example of a valid PyTorch model:
```python
class LogisticNet(nn.Module):
    def __init__(self, n_features, n_classes):
        super(LogisticNet, self).__init__()
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        _, x = x
        x = self.fc(x)
        return x
```
