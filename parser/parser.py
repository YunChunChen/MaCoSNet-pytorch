import argparse
from util.torch_util import str_to_bool


class ArgumentParser():

    def __init__(self, mode='train'):
        self.parser = argparse.ArgumentParser(description='PAMI implementation')
        self.add_base_parameters()
        self.add_cnn_model_parameters()

        if mode == 'train':
            self.add_train_parameters()
            self.add_loss_parameters()
            self.add_dataset_parameters()
            self.add_losses_parameters()

        elif mode == 'eval':
            self.add_eval_parameters()


    def add_base_parameters(self):
        base_params = self.parser.add_argument_group('base')
        # Image size
        base_params.add_argument('--image-size', type=int, default=240, help='image input size')
        # Pre-trained model file
        base_params.add_argument('--model', type=str, default='', help='Pre-trained model filename')
        base_params.add_argument('--model-aff', type=str, default='', help='Trained affine model filename')
        base_params.add_argument('--model-tps', type=str, default='', help='Trained TPS model filename')
        base_params.add_argument('--model-type', type=str, default='', help='Which model')
        # GPU
        base_params.add_argument('--gpu', type=int, default=0, help='gpu id')
        base_params.add_argument('--num-workers', type=int, default=8, help='number of workers')
    
        
    def add_dataset_parameters(self):
        dataset_params = self.parser.add_argument_group('dataset')
        # Image pair dataset parameters for train/val
        dataset_params.add_argument('--categories', nargs='+', type=int, default=0, help='indices of categories for training/eval')
        # Eval dataset parameters for early stopping
        dataset_params.add_argument('--eval-dataset', type=str, default='pf-pascal', help='Validation dataset used for early stopping')
        dataset_params.add_argument('--pck-alpha', type=float, default=0.1, help='pck margin factor alpha')
        dataset_params.add_argument('--eval-metric', type=str, default='pck', help='pck/distance')
        # Random synth dataset parameters
        dataset_params.add_argument('--random-crop', type=str_to_bool, nargs='?', const=True, default=True, help='use random crop augmentation')                
            

    def add_train_parameters(self):
        train_params = self.parser.add_argument_group('train')
        # Optimization parameters 
        train_params.add_argument('--lr', type=float, default=0.001, help='learning rate')
        train_params.add_argument('--momentum', type=float, default=0.9, help='momentum constant')
        train_params.add_argument('--num-epochs', type=int, default=10, help='number of training epochs')
        train_params.add_argument('--batch-size', type=int, default=16, help='training batch size')
        train_params.add_argument('--weight-decay', type=float, default=0, help='weight decay constant')
        train_params.add_argument('--seed', type=int, default=1, help='Pseudo-RNG seed')
        train_params.add_argument('--geometric-model', type=str, default='affine', help='geometric model to be regressed at output: affine or tps')
        # Trained model parameters
        train_params.add_argument('--result-model-dir', type=str, default='trained_models', help='path to trained models folder')
        # Dataset name (used for loading defaults)
        train_params.add_argument('--training-dataset', type=str, default='pf-pascal', help='dataset to use for training')
        # Parts of model to train
        train_params.add_argument('--train-fe', type=str_to_bool, nargs='?', const=True, default=True, help='Train feature extraction')
        train_params.add_argument('--train-fr', type=str_to_bool, nargs='?', const=True, default=True, help='Train feature regressor')
        train_params.add_argument('--train-bn', type=str_to_bool, nargs='?', const=True, default=True, help='train batch-norm layers')
        train_params.add_argument('--fe-finetune-params',  nargs='+', type=str, default=[''], help='String indicating the F.Ext params to finetune')
        train_params.add_argument('--update-bn-buffers', type=str_to_bool, nargs='?', const=True, default=False, help='Update batch norm running mean and std')
        train_params.add_argument('--self-correlation', type=str_to_bool, nargs='?', const=True, default=True, help='Compute self correlation')
        train_params.add_argument('--seg-mask', type=str_to_bool, nargs='?', const=True, default=True, help='Use segmentation mask')
        

    def add_loss_parameters(self):
        loss_params = self.parser.add_argument_group('loss')
        # Parameters of weak loss
        loss_params.add_argument('--tps-grid-size', type=int, default=3, help='tps grid size')
        loss_params.add_argument('--tps-reg-factor', type=float, default=0.2, help='tps regularization factor')
        loss_params.add_argument('--normalize-inlier-count', type=str_to_bool, nargs='?', const=True, default=True)
        loss_params.add_argument('--dilation-filter', type=int, default=0, help='type of dilation filter: 0:no filter;1:4-neighs;2:8-neighs')
        loss_params.add_argument('--use-conv-filter', type=str_to_bool, nargs='?', const=True, default=False, help='use conv filter instead of dilation')        

    def add_losses_parameters(self):
        losses_params = self.parser.add_argument_group('loss-param')
        # Loss parameters
        losses_params.add_argument('--match-loss', type=str_to_bool, nargs='?', const=True, default=False, help='use foreground guided matching loss?')        
        losses_params.add_argument('--w-match', type=float, default=0.0, help='weight for foreground guided matching loss')
        losses_params.add_argument('--cycle-loss', type=str_to_bool, nargs='?', const=True, default=False, help='use cycle consistency loss?')        
        losses_params.add_argument('--w-cycle', type=float, default=0.0, help='weight for cycle consistency loss')
        losses_params.add_argument('--trans-loss', type=str_to_bool, nargs='?', const=True, default=False, help='use transitive consistency loss?')  
        losses_params.add_argument('--w-trans', type=float, default=0.0, help='weight for transitive consistency loss')
        losses_params.add_argument('--coseg-loss', type=str_to_bool, nargs='?', const=True, default=False, help='use perceptual contrastive loss?')  
        losses_params.add_argument('--w-coseg', type=float, default=0.0, help='weight for perceptual contrastive loss')
        losses_params.add_argument('--task-loss', type=str_to_bool, nargs='?', const=True, default=False, help='use task consistency loss?')  
        losses_params.add_argument('--w-task', type=float, default=0.0, help='weight for task consistency loss')
        losses_params.add_argument('--grid-loss', type=str_to_bool, nargs='?', const=True, default=False, help='use transformed grid loss?')  
        losses_params.add_argument('--w-grid', type=float, default=0.0, help='weight for transformed grid loss')
        

    def add_eval_parameters(self):
        eval_params = self.parser.add_argument_group('eval')
        # Evaluation parameters
        eval_params.add_argument('--eval-dataset', type=str, default='pf-pascal', help='pf/caltech/tss')
        eval_params.add_argument('--flow-output-dir', type=str, default='results/', help='flow output dir')
        eval_params.add_argument('--pck-alpha', type=float, default=0.1, help='pck margin factor alpha')
        eval_params.add_argument('--eval-metric', type=str, default='pck', help='pck/distance')
        eval_params.add_argument('--tps-reg-factor', type=float, default=0.0, help='regularisation factor for tps tnf')
        eval_params.add_argument('--batch-size', type=int, default=16, help='training batch size')
        eval_params.add_argument('--log-dir', type=str, default='', help='log directory')
        eval_params.add_argument('--self-correlation', type=str_to_bool, nargs='?', const=True, default=False, help='Compute self correlation')
        

    def add_cnn_model_parameters(self):
        model_params = self.parser.add_argument_group('model')
        # Model parameters
        model_params.add_argument('--feature-extraction-cnn', type=str, default='resnet101', help='feature extraction CNN model architecture: vgg/resnet101')
        model_params.add_argument('--feature-extraction-last-layer', type=str, default='', help='feature extraction CNN last layer')
        model_params.add_argument('--fr-feature-size', type=int, default=15, help='image input size')
        model_params.add_argument('--fr-kernel-sizes', nargs='+', type=int, default=[7,5], help='kernels sizes in feat.reg. conv layers')
        model_params.add_argument('--fr-channels', nargs='+', type=int, default=[128,64], help='channels in feat. reg. conv layers')
        

    def parse(self, arg_str=None):

        if arg_str is None:
            args = self.parser.parse_args()

        else:
            args = self.parser.parse_args(arg_str.split())

        arg_groups = {}
        for group in self.parser._action_groups:
            group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
            arg_groups[group.title] = group_dict

        return (args, arg_groups)

        
