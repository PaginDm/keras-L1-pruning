# keras-L1-pruning

Prune simply model with your custom loss and optimizer. Prune only convs. 
Implement this scheme:

![picture](https://raw.githubusercontent.com/PaginDm/keras-L1-pruning/master/images/pruning.png)

## Todo list:
- [x] Convs L1-pruning by percent (https://openreview.net/pdf?id=rJqFGTslg)
- [x] Using with custom optimizers and losses 
- [x] Limit the pruning part by relative values
- [ ] Pruning flatten layer
- [ ] Jump to tensorflow>=2.0
- [ ] Pruning based on masked convs https://arxiv.org/abs/1512.08571

## Requirements
Tested with:
 - keras-surgeon==0.1.3
 - tensorflow==1.14
 - keras==2.3.1


## Edit the configuration file
```python
{
    "input_model_path": "model.h5",
    "output_model_path": "model_pruned.h5",
    "finetuning_epochs": 10, # the number of epochs for train between pruning steps
    "stop_loss": 0.1, # loss for stopping process
    "pruning_percent_step": 0.05 # part of convs for delete on every pruning step
    "pruning_standart_deviation_part": 0.2 # shift for limit pruning part
}

```

Explaining of params:
![picture](https://raw.githubusercontent.com/PaginDm/keras-L1-pruning/master/images/config_explain.png)

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

## Result
The model will be saved at "output_model_path". Besides, process will be saved as png (All values normalized to 1.0):

![picture](https://raw.githubusercontent.com/PaginDm/keras-L1-pruning/master/images/pruning_res.png)
