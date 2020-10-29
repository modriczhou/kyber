#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers

class ConvEncoder(layers.Layer):
    def __init__(self, a_num, c_num):
        super(ConvEncoder, self).__init__()
        self.embedding = layers.Embedding()

        self.conva_num = a_num
        self.convc_num = c_num

    def build(self, input_shape):
        super(ConvEncoder, self).build(input_shape)

    def call(self, input):
        pass


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        sum_list = []
        self.max_root_sum(root, sum_list)
        return max(sum_list)
    def max_root_sum(self, cur_root, sum_list):
        if not cur_root:
            return
        sum_list.append(self.get_max_cur(cur_root.left) + self.get_max_cur(cur_root.right) + cur_root.val)
        self.max_root_sum(cur_root.left, sum_list)
        self.max_root_sum(cur_root.right, sum_list)


    def get_max_cur(self, cur):
        if cur==None:
            return 0
        return max(self.get_max_cur(cur.left), self.get_max_cur(cur.right),0) + cur.val


class BSTIterator:
    def __init__(self, root: TreeNode):
        cur = root
        self.stack = []
        while cur:
            self.stack.append(cur)
            cur = cur.left

    def next(self) -> int:
        """
        @return the next smallest number
        """
        cur = self.stack.pop()
        res = self.cur.val

        if self.cur.right:
            self.cur = self.cur.right
            while (self.cur):
                self.stack.append(self.cur)
                self.cur = self.cur.left
            # return self.cur.val

        return res

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return bool(len(self.stack))

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def partition(num_list,cut):
            num_list[0], num_list[cut] = num_list[cut], num_list[0]
            i = 1
            j = len(nums)-1
            while True:
                while i < j and num_list[i]>nums[0]:
                    i+=1
                while j >=i and num_list[j]<=nums[0]:
                    j-=1
                if i>=j:
                    break
                num_list[i],num_list[j] = num_list[j],num_list[i]

            num_list[0], num_list[j] = num_list[j], num_list[0]
            return j
        cut = random.randint(0,len(nums)-1)
        mid_index = partition(nums, cut)
        if mid_index==k-1:
            return nums[mid_index]
        if mid_index<k-1:
            return self.findKthLargest(nums[mid_index+1:], k-mid_index)
        if mid_index>k-1:
            return self.findKthLargest(nums[:mid_index],k)



        


                
                














