class FunctionBackend(object):

    def __init__(self):
        self.function_classes = {}

    def __getattr__(self, name):
        fn = self.function_classes.get(name)
        if fn is None:
            raise NotImplementedError
        return fn

    def register_function(self, name, function_class):
        if self.function_classes.get(name):
            raise RuntimeError("Trying to register second function under name " + name + " in " + type(self).__name__)
        self.function_classes[name] = function_class



class THNNFunctionBackend(FunctionBackend):

    def __reduce__(self):
        return (_get_thnn_function_backend, ())

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __copy__(self):
        return self


def _get_thnn_function_backend():
    return backend


def _initialize_backend():
    from torch.nn._functions.thnn import _all_functions as _thnn_functions
    from layers.rnn import RNN, \
        RNNTanhCell, RNNReLUCell, GRUCell, LSTMCell
    from torch.nn._functions.dropout import Dropout, FeatureDropout

    backend.register_function('RNN', RNN)
    backend.register_function('RNNTanhCell', RNNTanhCell)
    backend.register_function('RNNReLUCell', RNNReLUCell)
    backend.register_function('LSTMCell', LSTMCell)
    backend.register_function('GRUCell', GRUCell)
    backend.register_function('Dropout', Dropout)
    backend.register_function('Dropout2d', FeatureDropout)
    backend.register_function('Dropout3d', FeatureDropout)
    for cls in _thnn_functions:
        name = cls.__name__
        backend.register_function(name, cls)


backend = THNNFunctionBackend()
_initialize_backend()
