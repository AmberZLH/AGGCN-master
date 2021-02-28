import torch
import numpy as np
test1 = torch.randn(2, 3)
#print(test1)
test2 = torch.zeros(3,4)
#print(test2)
test3 = torch.tensor([[2, 3, 4],[4, 5, 6]])
#print(test3)
s = test3.size()

"""
print(test3)
print(test4)
print(test3)
temp = test3.add_(10)
print(test3)
print(temp)
test4 = torch.tensor([[4,5,6], [7,8,9]])
print(test4)
n = torch.mm(test3, test4.t())
print(n)
n1 = test4*test3
print(n1) 
t1 = np.ones((2,3))
t2 = torch.from_numpy(t1)
print(t1)
print(t2)
t3 = t2.numpy()
print(t3) """
"""""weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = torch.nn.Embedding.from_pretrained(weight)
print(weight)
print(embedding)"""
"""x = np.array([[1, -2, -5, 9, 4, 3], [4,5,6,0,9,1]])
y = x.argsort()
print(y)"""
x = torch.tensor([1, -1, 4, 5, 0])
print(x)
print(torch.argsort(x))
print(torch.argsort(-x))