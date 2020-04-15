# -*- coding:utf-8 -*-


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


class Solution:
    # 生成链表
    def creat_list(self, array):
        head = ListNode(0)
        new_list = head
        for i in array:
            new_list.next = ListNode(i)
            new_list = new_list.next
        return head.next

    def print_list(self, head):
        while(head):
            print(head.val)
            head = head.next

    # 返回合并后列表
    def Merge(self, pHead1, pHead2):
        # write code here
        mergeHead = ListNode(90)
        p = mergeHead
        while pHead1 and pHead2:
            if pHead1.val >= pHead2.val:
                mergeHead.next = pHead2
                pHead2 = pHead2.next
            else:
                mergeHead.next = pHead1
                pHead1 = pHead1.next

            mergeHead = mergeHead.next
        if pHead1:
            mergeHead.next = pHead1
        elif pHead2:
            mergeHead.next = pHead2
        return p.next

    # 输入一个链表，反转链表后，输出新链表的表头。
    def ReverseList(self, pHead):
        n_list = None
        while(pHead):
            p_list = pHead.next
            l_list = pHead
            l_list.next = n_list
            n_list = l_list
            pHead = p_list
        return l_list

    # 输入一个链表，输出该链表中倒数第k个结点。
    def FindKthToTail(self, head, k):
        pass


array1 = [1, 3, 4, 5, 6, 7]

array2 = [2, 3, 4, 7, 8, 11, 13]

s = Solution()

n_list1 = s.creat_list(array2)

# n_list2 = s.creat_list(array2)

n_list1 = s.ReverseList(n_list1)

s.print_list(n_list1)
