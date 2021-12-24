import torch
import models.networks as networks
import util.util as util


class Ver2VerModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.net = torch.nn.ModuleDict(self.initialize_networks(opt))

    def forward(self, identity_points, pose_points, gt_points, id_face, mode):
        if mode == 'inference':
            pass
        else:
            identity_points=identity_points.transpose(2,1) #(bs, 3, n)
            identity_points=identity_points.cuda()
            
            pose_points=pose_points.transpose(2,1)  
            pose_points=pose_points.cuda()

            gt_points=gt_points.transpose(2,1)
            gt_points=gt_points.cuda()

        generated_out = {}
        if mode == 'train':
            
            loss, generated_out = self.compute_loss(identity_points, pose_points, gt_points, id_face)

            out = {}
            out['fake_points'] = generated_out['fake_points']
            out['identity_points'] = identity_points
            out['pose_points'] = pose_points
            out['warp_out'] = None if 'warp_out' not in generated_out else generated_out['warp_out']
            return loss, out

        elif mode == 'inference':
            out = {}
            with torch.no_grad():
                out = self.inference(identity_points, pose_points)
            out['identity_points'] = identity_points
            out['pose_points'] = pose_points
            return out
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list()
        G_params += [{'params': self.net['netG'].parameters(), 'lr': opt.lr}]
        G_params += [{'params': self.net['netCorr'].parameters(), 'lr': opt.lr}]

        beta1, beta2 = opt.beta1, opt.beta2 
        G_lr = opt.lr

        optimizer = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2), eps=1e-3)

        return optimizer

    def save(self, epoch):
        util.save_network(self.net['netG'], 'G', epoch, self.opt)
        util.save_network(self.net['netCorr'], 'Corr', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        net = {}
        net['netG'] = networks.define_G(opt)
        net['netCorr'] = networks.define_Corr(opt)

        if not opt.isTrain or opt.continue_train:
            net['netG'] = util.load_network(net['netG'], 'G', opt.which_epoch, opt)
            net['netCorr'] = util.load_network(net['netCorr'], 'Corr', opt.which_epoch, opt)

        return net

    def compute_loss(self, identity_points, pose_points, gt_points, id_face):
        losses = {}
        generate_out = self.generate_fake(identity_points, pose_points)
        
        # edge loss
        losses['edge_loss'] = 0.0
        for i in range(len(identity_points)):  
            f = id_face[i].cpu().numpy()
            v = identity_points[i].transpose(0,1).cpu().numpy()
            losses['edge_loss'] = losses['edge_loss'] + util.compute_score(generate_out['fake_points'][i].transpose(1,0).unsqueeze(0),f,util.get_target(v,f,1))
        losses['edge_loss'] = losses['edge_loss']/len(identity_points) * self.opt.lambda_edge

        # reconstruction loss
        losses['rec_loss'] = torch.mean((generate_out['fake_points'] - gt_points)**2) * self.opt.lambda_rec
        
        return losses, generate_out

    def generate_fake(self, identity_points, pose_points):
        generate_out = {}
        
        corr_out = self.net['netCorr'](pose_points, identity_points)
        generate_out['fake_points'] = self.net['netG'](corr_out['id_features'], corr_out['warp_out']) 

        generate_out = {**generate_out, **corr_out}
        return generate_out

    def inference(self, identity_points, pose_points):
        generate_out = {}

        corr_out = self.net['netCorr'](pose_points, identity_points)
        generate_out['fake_points'] = self.net['netG'](corr_out['id_features'], corr_out['warp_out'])

        generate_out = {**generate_out, **corr_out}
        return generate_out

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0