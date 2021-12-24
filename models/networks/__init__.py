import torch
from models.networks.base_network import BaseNetwork
from models.networks.generator import *
from models.networks.correspondence import *
import util.util as util

def find_network_using_name(target_network_name, filename, add=True):
    #for netG: elaingenerator
    target_class_name = target_network_name + filename if add else target_network_name  
    module_name = 'models.networks.' + filename
    network = util.find_class_in_module(target_class_name, module_name)

    assert issubclass(network, BaseNetwork), \
       "Class %s should be a subclass of BaseNetwork" % network

    return network

def modify_commandline_options(parser, is_train):
    opt, _ = parser.parse_known_args()

    netG_cls = find_network_using_name(opt.netG, 'generator')
    parser = netG_cls.modify_commandline_options(parser, is_train)

    return parser

def create_network(cls, opt, stage1=False):
    if stage1:
        net = cls(opt, stage1=True)
    else:
        net = cls(opt)
    net.print_network()
    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.cuda()
    net.init_weights(opt.init_type, opt.init_variance)
    return net

def define_G(opt):
    netG_cls = find_network_using_name(opt.netG, 'generator') #ElaINGenerator
    return create_network(netG_cls, opt)

def define_Corr(opt):  #Correspondence
    netCoor_cls = find_network_using_name('', 'correspondence')
    return create_network(netCoor_cls, opt)
