import keras.backend as K
import numpy as np

PI=3.14

def get_activations(model, inputs, print_shape_only=False, layer_name=None):
    # Documentation is available online on Github at the address below.
    # From: https://github.com/philipperemy/keras-visualize-activations
    print('----- activations -----')
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations


def get_data(n, input_dim, attention_column=1):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column] = y[:, 0]
    return x, y


def get_data_recurrent_indicator(n, time_steps, input_dim, attention_column=13):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y

def get_data_recurrent_sin(n, time_steps, input_dim, attention_column=13):
    """
    Data generation. x is purely random except that it's first value equals the target y.
    In practice, the network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    :param n: the number of samples to retrieve.
    :param time_steps: the number of time steps of your series.
    :param input_dim: the number of dimensions of each element in the series.
    :param attention_column: the column linked to the target. Everything else is purely random.
    :return: x: model inputs, y: model targets
    """
    print('Generating data ....')
    x = np.random.standard_normal(size=(n, time_steps, input_dim))   
    y = np.random.randint(low=0, high=2, size=(n, 1))
    
    freq = 0
    for i in range(n):        
        if y[i] ==0:
            freq = 0.1 * PI
        else:
            freq = 0.5 * PI
        for t in range(attention_column, int(attention_column+time_steps/2)):
            for d in range(input_dim):
                x[i, t, d] = np.sin(t*freq) + 0.05*np.random.randn(1)
    print('Done ....')                
    return np.asarray(x, 'float32'), np.asarray(y, 'float32')

# Using the generator pattern (an iterable)
class generator_recurrent_sin(object):
    def __init__(self, n, time_steps, input_dim, attention_column=13):
        self.n = n        
        self.time_steps = time_steps
        self.input_dim = input_dim        
        self.attention_column = attention_column
        self.num = 0
        
    def __iter__(self):
        return self.next()

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        if self.num < self.n:
            self.num += 1
            x = np.random.standard_normal(size=(self.time_steps, self.input_dim))   
            y = np.random.randint(low=0, high=2, size=(1))
         
            freq = 0
            if y ==0:
                freq = 0.1 * PI
            else:
                freq = 0.5 * PI
            for t in range(self.attention_column, int(self.attention_column+self.input_dim/2)):
                for d in range(self.attention_column):
                    x[t, d] = np.sin(t*freq) + 0.05*np.random.randn(1)
            yield x, y
        else:
            raise StopIteration()
    
get_data_recurrent = get_data_recurrent_sin