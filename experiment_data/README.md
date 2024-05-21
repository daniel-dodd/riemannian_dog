# Experiment data 📊.

This submodule contains the datasets used in the experiments. There are two main categories of datasets `pca` and `wordnet`. We import examples of each in the snippet below.
```python
from experiment_data.pca import iris
from experiment_data.wordnet import mammal_relations
```

## PCA:
These are PCA problems with labels `y` and features `X`. Additionally we have the `num_components` of the PCA problem, the `minimiser` and `minimum_loss` of the PCA problem solved by the `sklearn` implementation.
```python
iris()
```
```
Iris(X=Array([[-9.00681198e-01,  1.03205717e+00, -1.34127235e+00,
        -1.31297672e+00],
       [-1.14301693e+00, -1.24957599e-01, -1.34127235e+00,
        -1.31297672e+00],
       [-1.38535261e+00,  3.37848336e-01, -1.39813817e+00,
        -1.31297672e+00],
       [-1.50652051e+00,  1.06445365e-01, -1.28440666e+00,
        -1.31297672e+00],
       [-1.02184904e+00,  1.26346016e+00, -1.34127235e+00,
        -1.31297672e+00],
       [-5.37177563e-01,  1.95766914e+00, -1.17067528e+00,
        -1.05003083e+00],
       [-1.50652051e+00,  8.00654233e-01, -1.34127235e+00,
        -1.18150377e+00],
       [-1.02184904e+00,  8.00654233e-01, -1.28440666e+00,
        -1.31297672e+00],
       [-1.74885631e+00, -3.56360555e-01, -1.34127235e+00,
        -1.31297672e+00],
       [-1.14301693e+00,  1.06445365e-01, -1.28440666e+00,
        -1.44444966e+00],
       [-5.37177563e-01,  1.49486315e+00, -1.28440666e+00,
        -1.31297672e+00],
       [-1.26418483e+00,  8.00654233e-01, -1.22754097e+00,
        -1.31297672e+00],
       [-1.26418483e+00, -1.24957599e-01, -1.34127235e+00,
...
       [ 6.86617941e-02, -1.24957599e-01,  7.62758672e-01,
         7.90590823e-01]], dtype=float32), y=None, num_components=1, minimiser=array([[ 0.52237162],
       [-0.26335492],
       [ 0.58125401],
       [ 0.56561105]]), minimum_loss=Array(1.0891819, dtype=float32))
```

## Wordnet:
Field `X` (alias is `pairs`) encodes the relationships between the `ids`.
```python
mammal_relations()
```
```
MammalRelations(X=Array([[  0,   1],
       [  2,   3],
       [  4,   5],
       ...,
       [495,   9],
       [159, 121],
       [588, 669]], dtype=int32), y=None, mammal_ids=('flying_mouse', 'flying_phalanger', 'dairy_cattle', ...))
```
