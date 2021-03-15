import numpy as np
import os

from image import Image


if __name__ == '__main__':
	# get list of image names
	im_list = [file for file in os.listdir('./imgs') if file.endswith('.jpg')]

	# for each image, instantiate an Image object to calculate the Homography that map points from plane to image
	images = [Image(os.path.join('./imgs', im_file)) for im_file in im_list]

	# TODO: construct V to solve for b by stacking the output of im.construct_v() (Equation.(17))
	V = None

	# TODO: find b using the SVD trick
	b = None
	print('norm(V @ b) =', np.linalg.norm(V @ b))  # check if the dot product between V and b is zero
	b11, b12, b22, b13, b23, b33 = b.tolist()
	print('b.shape: ', b.shape)
	print('b11: ', b11)
	print('b12: ', b12)
	print('b22: ', b22)
	print('b13: ', b13)
	print('b23: ', b23)
	print('b33: ', b33)

	# TODO: find components of intrinsic matrix from Equation.(12)
	v0 = 0
	alpha = 0
	beta = 0
	c = 0
	u0 = 0

	print('----\nCamera intrinsic parameters:')
	print('\talpha: ', alpha)
	print('\tbeta: ', beta)
	print('\tc: ', c)
	print('\tu0: ', u0)
	print('\tv0: ', v0)
	cam_intrinsic = np.array([
		[alpha, c, u0],
		[0, beta, v0],
		[0, 0, 1]
	])

	# get camera pose
	for im in images:
		R, t = im.find_extrinsic(cam_intrinsic)
		print('R = \n', R)
		# print('R @ R.T = \n', R @ R.T)
		print('t = ', t)
