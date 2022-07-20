"""
The goal of this task is to stitch two images of overlap into one image.
You are given 'left.jpg' and 'right.jpg' for your image stitching code testing. 

"""

import cv2
import numpy as np
# np.random.seed(<int>) # you can use this line to set the fixed random seed if you are using np.random
import random
random.seed(5) # you can use this line to set the fixed random seed if you are using random


def solution(left_img, right_img):
    """
    :param left_img:
    :param right_img:
    :return: the result panorama image which is stitched by left_img and right_img
    """
    left_image_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_image_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    sift = cv2.xfeatures2d_SIFT.create()
    
    # finding the key points and descriptors with SIFT
    kp_right, des_right = sift.detectAndCompute(right_image_gray, None)
    kp_left, des_left = sift.detectAndCompute(left_image_gray, None)
    
    #performing KNN - finding two nearest neighbors
    print("performing KNN..........")
    table={}
    for i in range(len(des_right)):
        diff = des_left-des_right[i]
        dist=np.apply_along_axis(np.linalg.norm,1,diff)
        idx = np.argpartition(dist, 2)
        min1,min2=dist[idx[:2]]
        idx1,idx2=idx[:2]
        d= {'min1':min1,'idx1':idx1,'min2':min2,'idx2':idx2}
        table[i]=d
    
    print("performing ratio test..........")
    #Ratio Test
    good_matches_dict = {}
    for i in range(len(table)):
        if table[i]['min1'] < 0.75 * table[i]['min2']:
            good_matches_dict[i]=table[i]
            
            
    src_points=[]
    dest_pts=[]
    for key,values in good_matches_dict.items():
        sx,sy=kp_right[key].pt
        dx,dy=kp_left[good_matches_dict[key]['idx1']].pt
        src_points.append([sx,sy])
        dest_pts.append([dx,dy])
        
        
    #RANSAC Algorithm
    print("RANSAC Algorithm Started..........")
    num_sample=len(src_points)
    threshold=5.0
    iteration=1000
    sample_size=4
    bestH=None
    
    inlier_max=0
    inlier_list_index=[]
    for it in range(iteration):
        sampleIndex=random.sample(range(num_sample), sample_size)
        src=[src_points[i] for i in sampleIndex]
        dest=[dest_pts[i] for i in sampleIndex]
        
        H = homography(src,dest)
        inlier_index=[]
        inlier=0
        for i in range(len(src_points)):
            if i not in sampleIndex:
                s1 = src_points[i].copy()
                s1.append(1)
                s1=np.asarray(s1)
                d_calc = H@s1.T
                d_calc = (1/d_calc[-1])*d_calc
                d_orig = dest_pts[i].copy()
                dist = np.linalg.norm(d_calc[:2] - d_orig)
                if dist<threshold:
                    inlier_index.append(i)
                    inlier+=1
        if inlier>inlier_max:
            inlier_max=inlier
            inlier_list_index=inlier_index

    best_src=[src_points[i] for i in inlier_list_index]
    best_dst=[dest_pts[i] for i in inlier_list_index]
    best_H = homography(best_src,best_dst)

    
    result_img = StitchImages(left_img,right_img,best_H)
    print("Stitching Finished..........")
    return result_img

def homography(src,dest):
    #creating A matrix
    A=[]
    for i in range(len(src)):
        xs=src[i][0]
        ys=src[i][1]
        xd=dest[i][0]
        yd=dest[i][1]
        arr1=[xs,ys,1,0,0,0,-1*xd*xs,-1*xd*ys,-1*xd]
        arr2=[0,0,0,xs,ys,1,-1*yd*xs,-1*yd*ys,-1*yd]
        A.append(arr1)
        A.append(arr2)
    U,s,V = np.linalg.svd(A)
    M=V[-1]
    H=M.reshape(3,3)
    H = (1/M[-1]) * H
    return H

def StitchImages(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts2_Tr = cv2.perspectiveTransform(pts2, H)
    pts = np.concatenate((pts1, pts2_Tr), axis=0)
    
    [xmin, ymin] = np.int32(np.amin(pts,axis=0).reshape(-1))
    [xmax, ymax] = np.int32(np.amax(pts,axis=0).reshape(-1))
    
    translatex= -1*xmin
    translatey = -1*ymin

    translate_matrix = np.array([[1,0,translatex],[0,1,translatey],[0,0,1]]) 

    result = cv2.warpPerspective(img2, translate_matrix.dot(H), (xmax-xmin, ymax-ymin))
    result[translatey:h1+translatey,translatex:w1+translatex] = img1
    
    return result

if __name__ == "__main__":
    left_img = cv2.imread('left.jpg')
    right_img = cv2.imread('right.jpg')
    result_img = solution(left_img, right_img)
    cv2.imwrite('results/stich_result.jpg', result_img)


