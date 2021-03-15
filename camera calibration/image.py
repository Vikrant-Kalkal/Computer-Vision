'''
Define image class which read the image, extract chessboard corners find the homography
'''
import cv2
import numpy as np

objpoints = []  # 3d point in real world space

def normalize_trans(points):
	"""TODO: Compute a transformation which translates and scale the inputted points such that
		their center is at the origin and their average distance to the origin is sqrt(2) using Equation.(21)

	Args:
		points (np.ndarray): points to normalize, shape (n, 2)
	Return:
		np.ndarray: similarity transformation for normalizing these points, shape (3, 3)
	"""
	x = [p[0] for p in points]
	y = [p[1] for p in points]
	u = sum(x) / len(points)
	v =	sum(y) / len(points)

	dist = [np.sqrt( (p[0])**2 + (p[1])**2 ) for p in points]
	s=sum(dist)/len(points)

	T=np.array([s, 0, -s*u,0, s, -s*v,0, 0, 1]).reshape(3,3)

	return T

def homogenize(points):
	"""Convert points to homogeneous coordinate

	Args:
		points (np.ndarray): shape (n, 2)
	Return:
		np.ndarray: points in homogeneous coordinate (with 1 padded), shape (n, 3)
	"""
	re = np.ones((points.shape[0], 3))  # shape (n, 3)
	re[:, :2] = points
	return re


class Image:
	"""Provide operations on image necessary for calibration"""
	refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	def __init__(self, impath, square_size=0.03):
		"""
		Args:
			impath (str): path to image file
			square_size (float): size in meter of a square on the chessboard
		"""
		self.im = cv2.imread(impath)
		self.square_size = 0.03
		self.rows = 8  # number of rows in the grid pattern to look for on the chessboard
		self.cols = 6  # number of columns in the grid pattern to look for on the chessboard
		self.im_pts = self.locate_landmark()  # pixel coordinate of chessboard's corners
		self.plane_pts = self.get_landmark_world_coordinate()  # world coordinate of chessboard's corners
		self.H = self.find_homography()
		self.V = self.construct_v()

	def locate_landmark(self, draw_corners=False):
		"""Identify corners on the chessboard such that they form a grid defined by inputted parameters

		Args:
			draw_corners (bool): to draw corners or not
		Return:
			np.ndarray: pixel coordinate of chessboard's corners, shape (self.rows * self.cols, 2)
		"""
		# convert color image to gray scale
		gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
		# find the chessboard corners
		ret, corners = cv2.findChessboardCorners(gray, (self.rows, self.cols), None)
		# if found, refine these corners' pixel coordinate & store them
		if ret:
			corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), Image.refine_criteria)

			# self.im_pts = corners.squeeze()
			# print('self.im_pts.shape: ', self.im_pts.shape)
			# for i in range(self.im_pts.shape[0]):
			# 	print('pts [{}]: '.format(i), self.im_pts[i])

			if draw_corners:
				cv2.drawChessboardCorners(self.im, (self.rows, self.cols), corners, ret)
				cv2.imshow('im', self.im)
				cv2.waitKey(0)
				cv2.destroyWindow('im')

		return corners.squeeze()

	def get_landmark_world_coordinate(self):
		"""TODO: Compute 3D coordinate for each chessboard's corner. Assumption:

				* world origin is located at the 1st corner
				* x-axis is from corner 0 to corner 1 till corner 7,
				* y-axis is from corner 0 to corner 8 till corner 40,
				* distance between 2 adjacent corners is self.square_size


		Returns:
			np.ndarray: 3D coordinate of chessboard's corners, shape (self.rows * self.cols, 2)
		"""
		x = np.linspace(0, 7,8)
		y = np.linspace(0, 5,6)

		for i in range(0,6):
			for j in range(0, 8):

					objpoints.append([x[j] , y[i]])

		self.im_pts=np.array(objpoints)
		self.plane_pts= self.im_pts * self.square_size
		print("pixel points",self.im_pts)
		print(self.im_pts.shape)

		print("world points", self.plane_pts)
		print(self.plane_pts.shape)

		return(self.plane_pts)

		pass

	def find_homography(self):
		"""TODO: Find the homography H that maps plane_pts to im_pts using Equation.(8)

		Return:
			np.ndarray: homography, shape (3, 3)
		"""
		# get the normalize transformation
		T_norm_im = normalize_trans(self.im_pts)
		T_norm_plane = normalize_trans(self.plane_pts)

		# normalize image points and plane points
		norm_im_pts = (T_norm_im @ homogenize(self.im_pts).T).T  # shape (n, 3)
		norm_plane_pts = (T_norm_plane @ homogenize(self.plane_pts).T).T  # shape (n, 3)

		# TODO: construct linear equation to find normalized H using norm_im_pts and norm_plane_pts
		Q1=[]
		for i in range(48):
			Q1.append(np.array([norm_plane_pts[i],np.zeros(3),-norm_im_pts[i][0]*norm_plane_pts[i],np.zeros(3),norm_plane_pts[i],-norm_im_pts[i][1]*norm_plane_pts[i]]).reshape(2,9))
		Q=np.array(Q1).reshape(96,9)

		# TODO: find normalized H as the singular vector Q associated with the smallest singular value
		U,D,V=np.linalg.svd(Q)
		H_norm = np.array(V[np.argmin(D)]).reshape(3,3)  #check
		print(H_norm,"H norm")

		# TODO: de-normalize H_norm to get H
		H = np.linalg.inv(T_norm_im)*H_norm*T_norm_plane
		return H

	def construct_v(self):
		"""TODO: Find the left-hand side of Equation.(16) in the lab subject

		Return:
			np.ndarray: shape (2, 6)
		"""
		H=self.H

		v12 = [H[0][0]*H[0][1],H[0][0]*H[1][1]+H[1][0]*H[0][1],H[1][0]*H[1][1],H[2][0]*H[0][1]+H[0][0]*H[2][1],H[2][0]*H[1][1]+H[1][0]*H[2][1],H[2][0]*H[2][1]]
		v11 = [H[0][0]*H[0][0], H[0][0]*H[1][0] + H[1][0]*H[0][0], H[1][0]*H[1][0], H[2][0]*H[0][0] + H[0][0]*H[2][0], H[2][0]*H[1][0] + H[1][0]*H[2][0], H[2][0]*H[2][0]]
		v22 = [H[0][1]*H[0][1], H[0][1]*H[1][1] + H[1][1]*H[0][1], H[1][1]*H[1][1], H[2][1]*H[0][1] + H[0][1]*H[2][1], H[2][1]*H[1][1] + H[1][1]*H[2][1], H[2][1]*H[2][1]]

		vdiff=[]

		zip_object = zip(v11, v22)
		for v11_i, v22_i in zip_object:
			vdiff.append(v11_i - v22_i)

		v = np.array([v12 ,vdiff])

		pass

	def find_extrinsic(self, K):
		"""TODO: Find camera pose w.r.t the world frame defined by the chessboard using the homography and camera intrinsic
			matrix using Equation.(18)

		Arg:
			K (np.ndarray): camera intrinsic matrix, shape (3, 3)
		Returns:
			tuple[np.ndarray]: Rotation matrix (R) - shape (3, 3), translation vector (t) - shape (3,)
		"""
		pass

