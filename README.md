# keras-L1-pruning

Prune simply model with your custom loss and optimizer. Prune only convs

## Edit the configuration file
```python
{
    "input_model_path": "model.h5",
    "output_model_path": "model_pruned.h5",
    "finetuning_epochs": 10, # the number of epochs for train between pruning steps
    "stop_loss": 0.1, # loss for stopping process
    "pruning_percent_step": 0.05 # part of convs for delete on every pruning step
}

```

## Use like this
```python
import pruning
from keras.optimizers import Adam
from keras.utils import Sequence

train_batch_generator = BatchGenerator...
score_batch_generator = BatchGenerator...

opt = Adam(lr=1e-4)
pruner = pruning.Pruner("config.json", "categorical_crossentropy", opt)

pruner.prune(train_batch, valid_batch)
```