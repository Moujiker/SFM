# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
from utils.bundleAjust import bundleAdjustment
from utils.dense import denseMatch, denseReconstruction, outputPly
from utils.fundamental import default, implementacionRansac
from utils.getPose import getPose
from utils.graph import createGraph, triangulateGraph, showGraph, visualizeDense
from utils.mergeGraph import mergeG, removeOutlierPts
from utils.paresDescript import getPairSIFT

#Creditos a % SFMedu: Structrue From Motion for Education Purpose
# % Written by Jianxiong Xiao (MIT License) el codigo se base en este

#合并graph
def mergeAllGraph(gL,imsize):
    graphMerged = gL[0]
    # merge de vistas parciales
    for i in range(len(gL) - 1):
        graphMerged = updateMerged(graphMerged, gL[i+1],imageSize)
    return graphMerged
def updateMerged(gA,gB,imsize):
    gt = mergeG(gA, gB)
    gt = triangulateGraph(gt, imsize)
    gt = bundleAdjustment(gt, False)
    gt = removeOutlierPts(gt, 10)
    gt = bundleAdjustment(gt)
    return gt

if __name__ == "__main__":

    #---------------------------参数设置-----------------------#
    '''
    #图像最大分辨率
    #图片路径
    #调试模式
    #输出ply文件名
    #有效文件格式
    '''
    maxSize = 640 
    carpetaImagenes = 'example/'
    debug = False
    outName = "jirafa"
    validFile = ['jpg','png','JPG'] 
    
    # 尝试获取焦距 //Intentar conseguir la distancia focal
    # TODO：添加计算该值应该与图像一起使用 对于480x640 焦距为4mm的图像
    f = 719.5459

                          
    # ---------------------------SET PARAMETERS
    algoMatrizFundamental = implementacionRansac
    listaArchivos = os.listdir(carpetaImagenes)
    listaImages = filter(lambda x : x.split('.')[-1] in validFile,listaArchivos )

    #上传图像
    listaImages = map(lambda x : cv2.imread(carpetaImagenes+x),listaImages)

    imageSize = listaImages[0].shape
    print "Dimensiones originales ",imageSize
    #todo:如果大于maxSize，请缩放图像
    if imageSize[0] > maxSize:
        print "Size image ",imageSize," max size ",maxSize
        #480 640 funciona
        listaImages = map(lambda x: np.transpose(cv2.resize(x,(640,480)),axes=[1,0,2]), listaImages)
        imageSize = listaImages[0].shape
        print "Result size ",imageSize

    #计算矩阵 K
    K = np.eye(3)
    K[0][0] = f
    K[1][1] = f

    graphList = []
    graphList = [0 for i in range(len(listaImages)-1)]
    #基于sift计算对应pair点
    #他们的计算公式为连续图像
    print "calculate SIFT"
    for i in range(len(listaImages)-1):
        keypointsA,keypointsB = getPairSIFT(listaImages[i],listaImages[i+1],show=debug)

        #计算基本矩阵或本质矩阵
        if type(keypointsA[0]) == np.ndarray:
            assert(len(keypointsA.shape) == 2)
            assert (len(keypointsB.shape) == 2)
            pointsA = keypointsA
            pointsB = keypointsB
        else:
            pointsA = np.array([(keypointsA[idx].pt) for idx in range(len(keypointsA))]).reshape(-1, 1, 2)
            pointsB = np.array([(keypointsB[idx].pt) for idx in range(len(keypointsB))]).reshape(-1, 1, 2)
        pointsA = pointsA[:,[1,0]]
        pointsB = pointsB[:, [1, 0]]

        F = np.array(algoMatrizFundamental(pointsA,pointsB))
        Fmat = F[0]
        K = np.array(K)
        E = np.dot(np.transpose(K),np.dot(Fmat,K))

        # 获取相机的姿态
        Rtbest = getPose(E,K, np.hstack([pointsA,pointsB]),imageSize)

        print "F:", F
        print 'K:', K
        print 'E:', E
        print 'Rtbest:', Rtbest
        cv2.waitKey()
        #创建图
        graphList[i] = createGraph(i,i+1,K, pointsA, pointsB, Rtbest, f)

        #三角化
        graphList[i] = triangulateGraph(graphList[i],imageSize)

        #查看图像
        # showGraph(graphList[i],imageSize)

        #Bundle ajustement调整
        graphList[i]=bundleAdjustment(graphList[i])

        #可视化改进
        # showGraph(graphList[i], imageSize)

    gM = mergeAllGraph(graphList,imageSize)
    print "Merge Graph completed!!!"
    
    #查看部分结果
    showGraph(gM,imageSize)
    #密集匹配
    for i in range(len(listaImages)-1):
        graphList[i] = denseMatch(graphList[i],listaImages[i],
                                  listaImages[i+1], imageSize, imageSize)

    print "Dense match finished"
    print "Initializing dense Triangulation"
    #Dense reconstruction
    for i in range(len(listaImages) - 1):
        graphList[i] = denseReconstruction(graphList[i], gM,K,imageSize)

    print "Dense reconstruct finished"
    data = visualizeDense(graphList, gM, imageSize)

    outputPly(data,outName)
