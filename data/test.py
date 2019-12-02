import coviexinfo
import matplotlib.pyplot as plt
import numpy as np
import os
'''
extract('BasketballDrill.avi',representation,num_frames)
representation==0 denotes extracting mvs and ref.
representation==1 denotes extracting depth and qp.
representation==2 denotes extracting residual Y.
representation==3 denotes extracting residual U and V.
'''

#example 1, extracting mvs and ref.
os.chdir('/home/sjhu/datasets')
video = 'test.mp4'
num_frames=coviexinfo.get_num_frames('test.mp4')
#mv_ref_arr=(H/4,W/4,frames*6)
#mv_ref_arr is a array with 3 dimensions. The first dimension denotes Height of a frame. The second dimension denotes Width of a frame.
#For every frame, it contains mv_0_x, mv_0_y, ref_0, mv_1_x, mv_1_y, ref_1. So, the third dimension denote frames*6.
mv_ref_arr=coviexinfo.extract(video,0,num_frames)
f=mv_ref_arr[:,:,::6]
print(f.shape)
print(mv_ref_arr.shape)

#example 2, extracting depth and qp.
#depth_qp_arr=(H/16,W/16,frames*2),for every frame, it contains depth and qp.
depth_qp_arr=coviexinfo.extract(video,1,num_frames)
print(depth_qp_arr.shape)

#example 3, extracting residual Y.
#res_Y_arr=(H,W,frames)
res_Y_arr=coviexinfo.extract(video,2,num_frames)
print(res_Y_arr.shape)

#examples 4, extracting residual U and V.
#res_UV_arr=(H/2,W/2,frames*2), for every frame, it contains residual U and residual V.
res_UV_arr=coviexinfo.extract(video,3,num_frames)
print(res_UV_arr.shape)