import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

from utils import loadData, saveData, plotSamples, plotClassDist

class ResNet():
    
    def __init__(self, load = False, directory = './classification_model_ResNet34/batch_size32/', input_shape = None, num_classes = None,
                 num_filters = 32, kernel = 3, blocks = [3, 4, 6, 3]):
        ''''''
        if load:
            self.model = load_model(directory + 'resnet34_subject30_loss.h5')
            self.current_epoch = np.genfromtxt(directory + 'resnet34_hist_subject30_loss.csv', delimiter = ',', skip_header = 0).shape[0] - 1
            self.model.summary()
        else:
            assert (input_shape is not None or num_classes is not None), 'The arguments input_shape and num_filters must be specified if not loading a model.'
            self.model = self.build_model(input_shape, num_classes, num_filters, kernel, blocks)
            self.current_epoch = 0
    
    def build_model(self, input_shape, num_classes, num_filters, kernel, blocks):
        ''''''
        def residual_block(x, num_filters, kernel, stride, index):
            out = Conv2D(num_filters, kernel, stride, padding = 'same', name = 'C{}'.format(index))(x)
            index += 1
            out = ReLU(name = 'RELU{}'.format(index-1))(out)
            out = BatchNormalization(name = 'BN{}'.format(index-1))(out)
            out = Conv2D(num_filters, kernel, 1, padding = 'same', name = 'C{}'.format(index))(out)
            index += 1
            if stride == 2: #Need to downsample input by half to match dimensions
                x = Conv2D(num_filters, 1, stride, padding = 'same', name = 'C_DS{}'.format(index-3))(x)
            out = Add(name = 'ADD{}'.format(index-3))([x, out])
            out = ReLU(name = 'RELU{}'.format(index-1))(out)
            out = BatchNormalization(name = 'BN{}'.format(index-1))(out)
            return out, index
        
        def try_count_flops(model: Union[tf.Module, tf.keras.Model],
                    inputs_kwargs: Optional[Dict[str, Any]] = None,
                    output_path: Optional[str] = None):
            """
            Counts and returns model FLOPs.
            Args:
            model: A model instance.
            inputs_kwargs: An optional dictionary of argument pairs specifying inputs'
            shape specifications to getting corresponding concrete function.
            output_path: A file path to write the profiling results to.
            Returns:
            The model's FLOPs.
            """
            if hasattr(model, 'inputs'):
                try:
                    # Get input shape and set batch size to 1.
                    if model.inputs:
                        inputs = [
                            tf.TensorSpec([1] + input.shape[1:], input.dtype)
                            for input in model.inputs
                        ]
                        concrete_func = tf.function(model).get_concrete_function(inputs)
                    # If model.inputs is invalid, try to use the input to get concrete
                    # function for model.call (subclass model).
                    else:
                        concrete_func = tf.function(model.call).get_concrete_function(
                            **inputs_kwargs)
                    frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

                    # Calculate FLOPs.
                    run_meta = tf.compat.v1.RunMetadata()
                    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
                    if output_path is not None:
                        opts['output'] = f'file:outfile={output_path}'
                    else:
                        opts['output'] = 'none'
                    flops = tf.compat.v1.profiler.profile(
                        graph=frozen_func.graph, run_meta=run_meta, options=opts)
                    return flops.total_float_ops
                except Exception as e:  # pylint: disable=broad-except
                    logging.info(
                        'Failed to count model FLOPs with error %s, because the build() '
                        'methods in keras layers were not called. This is probably because '
                        'the model was not feed any input, e.g., the max train step already '
                        'reached before this run.', e)
                    return None
            return None

        x = Input(shape = input_shape, name = 'input')
        
        out = Conv2D(num_filters, kernel, 1, padding = 'same', name = 'C1')(x)
        out = ReLU(name = 'RELU1')(out)
        out = BatchNormalization(name = 'BN1')(out)
        
        index = 2
        for i in range(len(blocks)):
            for j in range(blocks[i]):
                if (i!=0 and j==0): stride = 2
                else: stride = 1
                out, index = residual_block(out, num_filters, kernel, stride, index)
            num_filters *= 2
                
        out = AveragePooling2D(name = 'avgPool')(out)
        out = Flatten(name = 'flatten')(out)
        out = Dense(num_classes, activation='softmax', name = 'output')(out)
        
        model = Model(x, out, name = 'resnet')
        flops = try_count_flops(model)
        print(flops/1000000,"M Flops")
        model.summary()
        return model
    
    def fit(self, X, Y, validation_data = None, epochs = 10, batch_size = 32, optimizer = 'adam',
            save = False, directory = './classification_model_DDcGAN/batch_size32/'):
        ''''''
        if save: # set callback functions if saving model
            if not os.path.exists(directory): os.makedirs(directory)
            mpath = directory + 'resnet18_DDcGAN_subject30_loss.h5'
            hpath = directory + 'resnet18_hist_DDcGAN_subject30_loss.csv'
            if validation_data is None:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'loss', verbose = 0, save_best_only = True)
            else:
                checkpoint = ModelCheckpoint(filepath = mpath, monitor = 'val_loss', verbose = 0, save_best_only = True)
            cvs_logger = CSVLogger(hpath, separator = ',', append = True)
            callbacks = [cvs_logger, checkpoint]
        else:
            callbacks = None
        
        self.model.compile(optimizer, 'categorical_crossentropy', ['accuracy']) #compile model
        self.model.fit( X, Y, validation_data = validation_data,
                        epochs = epochs, batch_size = batch_size, shuffle = True,
                        initial_epoch = self.current_epoch,
                        callbacks = callbacks,
                        verbose = 1 ) #fit model
                        
        self.current_epoch += epochs
        
    def predict(self, x):
        ''''''
        def entropy(p):
            return -1 * np.sum(np.log(p + 1e-15) * p, axis=0)
        
        pred = self.model.predict(x)
        entropy = np.apply_along_axis(entropy, axis = 1, arr = pred) 
        return pred, np.argmax(pred, axis = 1), entropy