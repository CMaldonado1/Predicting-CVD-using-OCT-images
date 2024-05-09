import yaml
from argparse import Namespace
from IPython import embed

def recursive_namespace(dd):
    '''
    Converts a (possibly nested) dictionary into a namespace.
    This allows for auto-completion
    '''
    for d in dd:
        has_any_dicts = False
        if isinstance(dd[d], dict):
            dd[d] = recursive_namespace(dd[d])
            has_any_dicts = True
    try:
      n =  Namespace(**dd)
      return n
    except:
      print(dd)


def load_config(yaml_config_file, args):
    
    
    with open(yaml_config_file) as config:
        config = yaml.safe_load(config)   
        #print("conf", config.network_architecture) 
        # I am using a namespace instead of a dictionary mainly because it enables auto-completion
    config = recursive_namespace(config)
    if args.w_kld is not None:
        config.w_kld = args.w_kld

    if args.latent_dim is not None:
        config.latent_dim = args.latent_dim

    if args.batch_size is not None:
        config.optimizer.batch_size = args.batch_size

    if args.lr is not None:
        config.optimizer.parameters.lr = args.lr
     
    if args.weight_decay is not None:
        config.optimizer.parameters.weight_decay = args.weight_decay

    if args.n_classes is not None:
        config.n_classes = args.n_classes

    if args.channels_class is not None:
        config.network_architecture.convolution.parameters.channels_class = args.channels_class

    if args.channels_concat is not None:
        config.network_architecture.convolution.parameters.channels_concat = args.channels_concat

    config.network_architecture.convolution.parameters.channels = \
    [int(x) for x in config.network_architecture.convolution.parameters.channels.split()]
#    config.network_architecture.convolution.parameters.channels_class = \
#    [int(x) for x in config.network_architecture.convolution.parameters.channels_class.split()]
#    config.network_architecture.convolution.parameters.channels_concat = \
#    [int(x) for x in config.network_architecture.convolution.parameters.channels_concat.split()]
    config.network_architecture.convolution.parameters.padding = \
    [int(x) for x in config.network_architecture.convolution.parameters.padding.split()]
    config.network_architecture.convolution.parameters.stride = \
    [int(x) for x in config.network_architecture.convolution.parameters.stride.split()] 
    config.network_architecture.convolution.parameters.kernel_size = \
    [int(x) for x in config.network_architecture.convolution.parameters.kernel_size.split()]
    config.input_dim = \
    [int(x) for x in config.input_dim.split()]
    
    return config
