import os
import torch
import numpy as np
import pymesh
from options.test_options import TestOptions
from models.ver2ver_model import Ver2VerModel

opt = TestOptions().parse()
   
torch.manual_seed(0)

model = Ver2VerModel(opt)
model.eval()

def face_reverse(faces, random_sample):
    identity_faces=faces
    face_dict = {}
    for i in range(len(random_sample)):
        face_dict[random_sample[i]] = i
    new_f = []
    for i in range(len(identity_faces)):
        new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
    new_face = np.array(new_f)
    return new_face

if opt.dataset_mode == 'human':
    print('test model on unseen data in SMPL')

    save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), opt.results_dir, 'human')
    test_list_name = 'human_test_list'
    vertex_num = 6890
elif opt.dataset_mode == 'animal':
    print('test model on unseen data in SMAL')

    save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), opt.results_dir, 'animal')
    test_list_name = 'animal_test_list'
    vertex_num = 3889
else:
    raise ValueError("|dataset_mode| is invalid")

if not os.path.exists(save_root):
            os.makedirs(save_root)

data_path = opt.dataroot
PMD_test = 0.0
mesh_num = 0
for line in open(test_list_name, "r"):
    mesh_num += 1
    data_list = line.strip('\n').split(' ')
    id_mesh_name = data_list[0]
    pose_mesh_name = data_list[1]
    gt_mesh_name = data_list[2]
    
    identity_mesh = pymesh.load_mesh(data_path + id_mesh_name)
    pose_mesh = pymesh.load_mesh(data_path + pose_mesh_name)
    gt_mesh = pymesh.load_mesh(data_path + gt_mesh_name) 

    random_sample = np.random.choice(vertex_num,size=vertex_num,replace=False)
    random_sample2 = np.random.choice(vertex_num,size=vertex_num,replace=False)

    identity_points = identity_mesh.vertices[random_sample]
    identity_points = identity_points - (identity_mesh.bbox[0] + identity_mesh.bbox[1]) / 2
    identity_points = torch.from_numpy(identity_points.astype(np.float32)).cuda()
    
    pose_points = pose_mesh.vertices[random_sample2]
    pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
    pose_points = torch.from_numpy(pose_points.astype(np.float32)).cuda()

    gt_mesh_points = gt_mesh.vertices[random_sample]
    gt_mesh_points = gt_mesh_points - (gt_mesh.bbox[0] + gt_mesh.bbox[1]) / 2

    # generate results
    out = model(identity_points.transpose(1,0).unsqueeze(0), pose_points.transpose(1,0).unsqueeze(0), None, None, mode='inference')

    out['fake_points'] = out['fake_points'].squeeze().transpose(1,0).cpu().detach().numpy()
    bbox = np.array([[np.max(out['fake_points'][:,0]),np.max(out['fake_points'][:,1]),np.max(out['fake_points'][:,2])],
                    [np.min(out['fake_points'][:,0]),np.min(out['fake_points'][:,1]),np.min(out['fake_points'][:,2])]])
    out['fake_points'] = out['fake_points'] - (bbox[0] + bbox[1] ) / 2
    
    # calculate PMD
    PMD_test = PMD_test + np.mean((out['fake_points'] - gt_mesh_points)**2)     

    # save the generated meshes 
    new_face = face_reverse(identity_mesh.faces, random_sample)    
    if opt.dataset_mode == 'human':
        pymesh.save_mesh_raw(save_root + '/' + gt_mesh_name, out['fake_points'], new_face) 
    elif opt.dataset_mode == 'animal':
        pymesh.save_mesh_raw(save_root + '/' + gt_mesh_name.strip('.ply') + '.obj', out['fake_points'], new_face) 

print('Final score for ' + test_list_name + ' is ' + str(PMD_test/mesh_num)) 
        