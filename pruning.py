from keras.models import load_model
import json
import os
from utils import prune_model, plot_and_save_stats
import time

class Pruner():
    def __init__(self, config_path, loss, optimizer):
        """
        Init pruner and compile model
        ``config_path`` Path to config file
        ``loss`` An instance of loss (keras.losses)
        ``optimizer`` An instance of optimizer (keras.optimizers)
        """

        with open(config_path) as config_buffer:    
            config = json.loads(config_buffer.read())

        self.loss = loss
        self.optimizer = optimizer
        self.input_model_path = config['input_model_path']
        self.output_model_path = config['output_model_path']
        self.finetuning_epochs = config["finetuning_epochs"]
        self.stop_loss = config["stop_loss"]
        self.pruning_percent_step = config["pruning_percent_step"]
        self.pruning_standart_deviation_part = config["pruning_standart_deviation_part"]
        self.stats = []

    def prune(self, train_generator, score_generator):
        """
        ``train_generator`` A generator or an instance of Sequence (keras.utils.Sequence) for finetuning model between pruning steps.
        ``score_generator`` A generator or an instance of Sequence (keras.utils.Sequence) for control stopping. Pruning will stop, when evaluate_generator(score_generator) return loss lower than self.config.stop_loss
        """
        pruned_model = load_model(self.input_model_path)
        pruned_model.compile(loss = self.loss, optimizer = self.optimizer)

        start = time.time()
        score_loss = pruned_model.evaluate_generator(score_generator)
        diff = time.time() - start

        stat = {'object': []}
        stat['size'] = os.path.getsize(self.input_model_path) / 1024 / 1024
        stat['loss'] = score_loss
        stat['time'] = diff

        self.stats.append(stat)
        print(f"Start. Size = {stat['size']} mb. Loss = {stat['loss'] }. Inference time: {stat['time']} sec")

        step = 0
        while(score_loss < self.stop_loss):            
            step+=1
            pruned_model = prune_model(pruned_model, self.pruning_percent_step, self.pruning_standart_deviation_part)
            pruned_model.compile(loss = self.loss, optimizer = self.optimizer)

            # finetuning
            pruned_model.fit_generator(
                generator = train_generator,
                epochs = self.finetuning_epochs
                )

            start = time.time()
            score_loss = pruned_model.evaluate_generator(score_generator)
            diff = time.time() - start

            if(score_loss < self.stop_loss):
                pruned_model.save(self.output_model_path)
                stat = {'object': []}
                stat['size'] = os.path.getsize(self.output_model_path) / 1024 / 1024
                stat['loss'] = score_loss
                stat['time'] = diff

                self.stats.append(stat)
                print(f"Step {step}. Size = {stat['size']} mb. Loss = {stat['loss']}. Inference time: {stat['time']} sec")

        # Create and save figure
        plot_and_save_stats(self.stats)









