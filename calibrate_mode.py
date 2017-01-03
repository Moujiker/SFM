#!/usr/bin/env python

import numpy as np
import cv2
import os
from common import splitfn

import sys, getopt
from glob import glob

USAGE = '''
USAGE: calib.py [--save <filename>] [--debug <output path>] [--square_size] [<image mask>]
'''
class camera_calibrate:
	def __init__(self,n):  
		self._pic_num = n  
		self._rms = 0  
		self._camera_mat = []
		self._dist_coefs  = []
		self._pattern_points = []
		self._img_points = []
		self._obj_points = []

	def run(self):
		args, img_mask = getopt.getopt(sys.argv[1:], '', ['save=', 'debug=', 'square_size='])
		args = dict(args)
		try: img_mask = img_mask[0]
		except: img_mask = './pic/left*.jpg'
		img_names = glob(img_mask)
		debug_dir = args.get('--debug')
		square_size = float(args.get('--square_size', 1.0))

		pattern_size = (9, 6)
		pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 )
		pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
		pattern_points *= square_size

		obj_points = []
		img_points = []
		h, w = 0, 0
		for fn in img_names:
		    print 'processing %s...' % fn,
		    img = cv2.imread(fn, 0)
		    h, w = img.shape[:2]
		    found, corners = cv2.findChessboardCorners(img, pattern_size)
		    if found:
		        term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
		        cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
		    if debug_dir:
		        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		        cv2.drawChessboardCorners(vis, pattern_size, corners, found)
		        path, name, ext = splitfn(fn)
		        cv2.imwrite('%s/%s_chess.bmp' % (debug_dir, name), vis)
		    if not found:
		        print 'chessboard not found'
		        continue
		    img_points.append(corners.reshape(-1, 2))
		    obj_points.append(pattern_points)

		    print 'ok'

		self._rms, self._camera_mat, self._dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h))
		print "RMS:", self._rms
		print "camera matrix:\n", self._camera_mat
		print "distortion coefficients: ", self._dist_coefs.ravel()
		cv2.destroyAllWindows()

if __name__ == '__main__':
	cb = camera_calibrate(14)
	cb.run()

