import torch.utils.data as data
import torch
import numpy as np
import pymesh

class SMPL_DATA(data.Dataset):
    def __init__(self, dataroot, vertex_num=6890, shuffle_point = True):
        self.shuffle_point = shuffle_point 
        self.vertex_num = vertex_num
        self.path= dataroot
        self.datapath = []
        for _ in range(4000):
            identity_i = np.random.randint(0,16)
            identity_p = np.random.randint(200,600)
            data_in = [identity_i, identity_p]
            self.datapath.append(data_in)

    def __getitem__(self, index):
        np.random.seed()
        mesh_set = self.datapath[index]
        identity_mesh_i = mesh_set[0]
        identity_mesh_p = mesh_set[1]
        pose_mesh_i = np.random.randint(0,16)
        pose_mesh_p = np.random.randint(200,600)
        identity_mesh = pymesh.load_mesh(self.path+'id'+str(identity_mesh_i)+'_'+str(identity_mesh_p)+'.obj')
        pose_mesh = pymesh.load_mesh(self.path+'id'+str(pose_mesh_i)+'_'+str(pose_mesh_p)+'.obj')
        gt_mesh = pymesh.load_mesh(self.path+'id'+str(identity_mesh_i)+'_'+str(pose_mesh_p)+'.obj')

        identity_points = identity_mesh.vertices
        identity_faces = identity_mesh.faces
        pose_points = pose_mesh.vertices
        pose_faces = pose_mesh.faces
        gt_points = gt_mesh.vertices
        
        pose_points = pose_points - (pose_mesh.bbox[0] + pose_mesh.bbox[1]) / 2
        pose_points = torch.from_numpy(pose_points.astype(np.float32))
        
        identity_points = identity_points-(identity_mesh.bbox[0]+identity_mesh.bbox[1])/2
        identity_points = torch.from_numpy(identity_points.astype(np.float32))

        gt_points = gt_points-(gt_mesh.bbox[0]+gt_mesh.bbox[1]) / 2
        gt_points = torch.from_numpy(gt_points.astype(np.float32))

        random_sample = np.random.choice(self.vertex_num,size=self.vertex_num,replace=False)
        random_sample2 = np.random.choice(self.vertex_num,size=self.vertex_num,replace=False)

        new_id_faces = identity_faces
        new_pose_faces = pose_faces

        # Before input, shuffle the vertices randomly to be close to real-world problems.
        if self.shuffle_point:
            pose_points = pose_points[random_sample2]
            identity_points = identity_points[random_sample]
            gt_points = gt_points[random_sample]
            
            face_dict = {}
            for i in range(len(random_sample)):
                face_dict[random_sample[i]] = i
            new_f = []
            for i in range(len(identity_faces)):
                new_f.append([face_dict[identity_faces[i][0]],face_dict[identity_faces[i][1]],face_dict[identity_faces[i][2]]])
            new_id_faces = np.array(new_f)

            face_dict = {}
            for i in range(len(random_sample2)):
                face_dict[random_sample2[i]] = i
            new_f = []
            for i in range(len(pose_faces)):
                new_f.append([face_dict[pose_faces[i][0]],face_dict[pose_faces[i][1]],face_dict[pose_faces[i][2]]])
            new_pose_faces = np.array(new_f)

        return identity_points, pose_points, gt_points, new_id_faces, new_pose_faces
        
    def __len__(self):
        return len(self.datapath)
