#!/usr/bin/env python
# coding: utf-8

# In[8]:


print("Hi,Bye")


# In[9]:


a=10
b=1
print(a==b)


# In[5]:


a=10
b=10
c=20
print(a==b or a>=c)


# In[22]:


l1=[1,2,3,4]
l1.remove(2)
print(2 in l1)


# In[29]:


l2=[1,2,3,4]
l3=[1,2,3,4]
print(l3==l2)


# In[32]:


a=0
print(bool(a))


# In[38]:


a=10
if a>50:
    print("a is greater than 50")
if a<50:
    print("a is less than 50")
print("end")


# In[45]:


a=100
b=10
if (a==b):
    print("a is equal to b")
elif (a>b):
    print("a is greater than b")
else:
    print("a is less than b")


# In[48]:


a=int(input("Enter the first no.:"))
b=int(input("Enter the second no.:"))
c=int(input("Enter the third no.:"))
if (a>b):
    if(a>c):
        print("a is largest")
    else:
        print("c is largest")
elif(b>a):
    if(b>c):
        print("b is largest")
    else:
        print("c is largest")


# In[ ]:


````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````


# In[64]:


total=0
while total<10:
    no=int(input("Enter the no.:"))
    total+=no
print(total)


# In[68]:


x=1
print("Loop Start")
while x<10:
    print(x)
    x+=2


# In[72]:


a=[1,2,3,40,5]
for i in a:
    print(i)


# In[76]:


for i in range(10):
    print(str(i)*i) #not i*i which will give us a line of squares!


# In[84]:


A=['Pen','Pencil','Book','Toy']
B=['School','College']
for i in A:
    print(i)
    for j in B:
        print(" ",j)
        break
        print(" ",j)


# In[85]:


def add(num1,num2):
    sum=num1+num2
    print(sum)


# In[92]:


add(4,2)


# In[93]:


def sale():
    quantity=int(input("Enter the Quantity:"))
    prince =int(input("Enter the Price:"))
    amt=quantity*prince
    print("The Amount is:",amt)


# In[94]:


sale()


# In[96]:


sale()


# In[114]:


def sale(qty=10,pr=100):
    amt=qty*pr
    print(amt)
    return amt


# In[105]:


sale(10,20)


# In[115]:


total_sales=[]
total_sales.append(sale(11,22))


# In[116]:


total_sales.append(sale(11,21))


# In[117]:


total_sales.append(sale(10,22))


# In[118]:


total_sales


# In[119]:


def fin(revenue,expense):
    profit=revenue-expense
    profit_ratio=profit/revenue
    new_fin=(profit,profit_ratio)
    return new_fin


# In[123]:


x=fin(1000,900)
print(x)


# In[124]:


print(x[0])


# In[125]:


def convertto(deg):
    f=(deg*9/5)+32
    return f


# In[127]:


convertto(38)


# In[21]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[ ]:





# In[25]:



from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6


# In[27]:


days=[1,2,3,4,5]
sleep=[7,8,9,10,11]
eat=[2,3,4,5,6,7]
work=[8,9,9,7,1]
play=[7,4,1,1,1]
plt.stackplot (days,sleep,eat,work,play)
plt.show()


# In[28]:


hrs=[8,2,7,3]
plt.pie(hrs)
plt.show()


# In[ ]:




