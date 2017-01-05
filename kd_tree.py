#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

class kd_node():
    def __init__(self,point = None, split = None, l_tree = None, r_tree = None):
        self._point = point
        self._split = split
        self._left_node = l_tree
        self._right_node = r_tree
        

class kd_tree():
    def __init__(self):
        self._data_list = None
        self._root = None
        self._data_len = 0
          
    def create_kd_tree(self,root, data_ls):

        self._data_list = data_ls
        data_list = data_ls
        self._data_len = len(data_list)
        LEN = self._data_len
        
        if LEN == 0:
            return
        dim = len(data_list[0])
        max_var = 0
        split = 0
        for i in range(dim):
            ll = []
            for data in data_list:
                ll.append(data[i])
            var = self.campute_Variance(ll)
            if var > max_var:
                max_var = var
                split = i
        data_list.sort(key=lambda x: x[split]) #根据划分域的数据对数据点进行排序 
        point = data_list[LEN / 2]  #选择下标为len / 2的点作为分割点  
        root = kd_node(point, split)  
        root._left_node = self.create_kd_tree(root._left_node, data_list[0:(LEN / 2)])  
        root._right_node = self.create_kd_tree(root._right_node, data_list[(LEN / 2 + 1):LEN])  
        return root  
                
    def campute_Variance(self,arrayList):
        for le in arrayList:
            le = float(le)
        LEN = len(arrayList)
        array = np.array(arrayList)  
        sum1 = array.sum()  
        array2 = array * array  
        sum2 = array2.sum()  
        mean = sum1 / LEN
        variance = sum2 / LEN - mean**2  
        return variance        
        
    def preorder(self, root):
        print root._point
        if root._left_node:  
            self.preorder(root._left_node)  
        if root._right_node:  
            self.preorder(root._right_node) 
    
    def midorder(self, root):
        pass
    def postorder(self, root):
        pass

def test():
    data_list = [[1,3],[4,2],[6,9],[3,5],[4,6],[2,7],[5,9],[5,1]]
    k_d = kd_tree()
    k_d._root = k_d.create_kd_tree(k_d._root, data_list)
    k_d.preorder(k_d._root)
        
if __name__ == "__main__":
    
    test()
