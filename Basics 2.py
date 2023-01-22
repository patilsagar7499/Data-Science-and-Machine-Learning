#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
print("Python Version:")
print(sys.version)
print("Version info:")
print(sys.version_info)


# In[6]:


import datetime
x=datetime.datetime.now()
print(x)


# In[7]:


from math import pi
r=int(input("Enter the radius:"))
Area=pi*r*r
print("Area is",str(Area))


# In[ ]:


Fname=input("Enter your first name:")
Lname=input("Enter your last name:")
print(Lname," ",Fname)


# In[ ]:


values=input("Enter some comma separated nos:")
list1=values.split(",")
tuple1=tuple(list1)
print("List",list1)
print("Tuple",tuple1)


# In[ ]:


l1=[10,20,30]
print(l1)
print(type(l1))


# In[ ]:


l1=[1,2,3,[4,5,6],'Sagar','VIIT']
print(l1)
print(type(l1))


# In[ ]:


len(l1)


# In[ ]:


l1[0]


# In[ ]:


l1[3][1]


# In[ ]:


l1[-1]


# In[ ]:


l1[-3:]


# In[ ]:


l1[-5:-2]


# In[ ]:


l1[1:7:2]


# In[ ]:


l2=[9,8,7]
l3=l1+l2
l3


# In[ ]:


l4=l3+[4,5,6]
l4


# In[ ]:


l4*3


# In[ ]:


(5 in l4)


# In[ ]:


11 in l4


# In[ ]:


del(l4[2])
l4


# In[ ]:


l4.clear()
l4


# In[ ]:


l4


# In[ ]:


l1=[10,20,30,40,50,60,70,80,90]
l1.remove(10)
l1


# In[ ]:


m=l1.pop(2)
l1


# In[ ]:


del(l1[2])
l1


# In[ ]:


l1.append(2)
l1


# In[ ]:


l1.extend([5,1])
l1


# In[ ]:


l1.insert(1,2)
l1


# l1[0]

# In[ ]:


l1[0]


# In[ ]:


l1[0]=10
l1


# In[ ]:


l1=[]
for x in [1,2,3,4,5]:
    l1.append(x*2)
print(l1)


# In[ ]:


l1=[]
for x in [1,2,3,4,5]:
    l1.append(x*2)
    print(l1)
print(2*l1)


# In[ ]:


l1=['sagar','Sagar','1','a','A','9']
sorted(l1)


# In[ ]:


l1=[1,12,3,4,5,6]
sorted(l1)


# In[ ]:


l1[::-1]


# In[ ]:


l1[4:2:-1]


# In[ ]:


l1[1]=2
l1


# In[ ]:


nums=[1,2,8,3,4]
x=max(nums)
y=sum(nums)
print(x)
print(y)


# In[ ]:


sq=[]
for i in range(1,10):
    s=i**2
    sq.append(s)
print(sq)
 


# In[ ]:


colors=[]
upp_colors=['a','s','d','f']
for x in upp_colors:
    f=2*x
    colors.append(f)
colors


# In[ ]:


t1=(1,2,3)
type(t1)


# In[ ]:


t1[0:4]*2


# In[ ]:


len(t1)


# In[ ]:


x='Good day'
'o'in x


# In[ ]:


x.find('a')


# In[ ]:


t1


# In[ ]:


set(t1)


# In[ ]:


x=[1,2,3]
x


# In[ ]:


x.append(1)
x


# In[ ]:


a='Python'
b='Training'
fee=5000
sl=100
print("%d -Welcome to %s %s and it's fee is %d"%(sl,a,b,fee))


# In[ ]:


a={1,2,3}
b={4,5,6,2}
a|b


# In[ ]:


a&b


# In[ ]:


a.intersection(b)


# In[ ]:


a.difference(b)


# In[ ]:


a.issubset(b)


# In[ ]:


b=frozenset(a)
print(b)
print(type(b))


# In[ ]:


a.add(10)
a


# In[ ]:


a=int(input("Enter the no.:"))
n1=int("%s"% a)
n2=int("%s%s"%(a,a))
n3=int("%s%s%s"%(a,a,a))
print(n1+n2+n3)


# In[ ]:


time=float(input("Enter the time in secs:"))
Day=time//(24*3600)
time%=24*3600
Hour=time//(3600)
time%=3600
minutes=time//(60)
time%=60
seconds=time
print("%d days:%d hours:%d minutes:%d seconds"%(Day,Hour,minutes,seconds))


# In[ ]:


def list_count_4(nums):
    count=0
    for num in nums:
        if num==4:
            count=count+1
            
    return count


# In[ ]:


print(list_count_4([4]))


# In[ ]:


def vowel(char):
    all_vowel='aeiou'
    return char in all_vowel


# In[ ]:


vowel('a')


# In[ ]:


def historogram(items):
    for n in items:
        output=''
        times=n
        while(times>0):
            output+='$'
            times=times-1
        print(output)


# In[ ]:


historogram([1,2,3,4,5])


# In[ ]:


emp={'Name':'Sagar','Age':20}


# In[ ]:


emp['Name'][2]


# In[ ]:


emp.items()


# In[ ]:


emp.copy()


# In[ ]:


Name=['Sagar','Amey','Rahul','Rohan']
Mark=[100,0,90,80]


# In[ ]:


for (c,o) in zip(Name,Mark):
    print(c,":",o)


# In[ ]:


import numpy as np


# In[ ]:


np.array([])


# In[ ]:


a=np.array([])
print(a)


# In[ ]:


type(a)


# In[ ]:


a.ndim


# In[ ]:


a=np.array([1,2,3,4,5])
print(a)


# In[ ]:


a.ndim


# In[ ]:


a.shape


# In[ ]:


l1=[1,2,3,4,5,'Sagar']
print(l1)
a=np.array(l1)
print(a)


# 

# In[ ]:


l1=[1,True,False,2]
print(l1)
a=np.array(l1)
print(a)


# In[ ]:


a=np.array([[1,2,3,4,5],[6,7,8,9,0]])
print(a)


# In[ ]:


a.ndim


# In[ ]:


a.shape


# In[ ]:


type(a)


# In[ ]:


print(a[0:])


# In[ ]:


a=np.array([1,2,3,4,5])
print(a[0:2])


# In[ ]:


print(np.arange(3))


# In[ ]:


a=np.array([[[1,2,3],[4,5,6]],[[7,8,9],[0,1,2]]])
a.ndim


# In[ ]:


a=np.array([[[1,2,3],[4,5,6]]])
print(a)
print(a.ndim)


# In[ ]:


a.shape


# In[ ]:


print(a)


# In[ ]:


a.reshape(3,2,2)


# In[ ]:


print(np.ones((5,5))*5)


# In[ ]:


a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a)
b=np.array([[2,3,1],[5,6,4],[8,9,7]])
print(b)
print(a@b)


# In[ ]:


print(b.ndim)
c=b.ravel()
print(c.ndim)


# In[ ]:


a=np.linspace(1,2,3)
print(a)


# In[ ]:


a=np.arange(10,20)
print(a)


# In[ ]:


b=a.reshape(2,5)
print(b)


# In[ ]:


c=np.delete(b,2,1)
print(c)


# In[ ]:


import pandas as pd


# In[ ]:


a=pd.Series()
print(type(a))


# In[ ]:


a=pd.Series([1,2,3])
print(a)


# In[ ]:


a=np.array([1,2,3])
pd.DataFrame(a)


# In[ ]:


b[1]


# In[ ]:


l1=[{'Name':'Sagar','Age':20},{'Name':'xyz','Age':123},{'Name':'dkjsh','Age':2},{'Name':'xy','Age':12}]
pd.DataFrame(l1)


# In[ ]:


a=[['Sagar','Good'],['kjdkj',341265],[847,4498]]
b=pd.DataFrame(a)
print(b)


# In[ ]:


x=int(input("Enter:"))
print(x)


# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


l1=[10,20,30,40,50]
pd.DataFrame(l1)


# In[ ]:


l2=[{'Name':'Sagar','Age':19},{'Name':'Amey','Age':20},{'Name':'Swapnil','Age':21},{'Name':'Pranav','Age':22}]
pd.DataFrame(l2)


# In[ ]:


l3={'Name':['Sagar','Amey','Swapnil','Pranav'],'Age':[19,20,21,22]}
pd.DataFrame(l3)


# In[ ]:


l3['Name']
print(l3)
pd.DataFrame(l3['Age'])


# In[ ]:


l1=[10,20,30,40,50]
pd.DataFrame(l1,index=['a','b','c','d','e'])


# In[ ]:


l3={'Name':['Sagar','Amey','Swapnil','Pranav'],'Age':[19,20,21,22]}
pd.DataFrame(l3)


# In[ ]:


pd.DataFrame(l3).loc[1]


# In[ ]:


pd.DataFrame(l3).iloc[1:3,0:2]


# In[ ]:


pd.DataFrame(l3).append(l2)


# In[ ]:


pd.DataFrame(l3).reset_index()


# In[ ]:


x=pd.DataFrame(l3).reset_index()
x


# In[ ]:


del(x['index'])


# In[ ]:


x


# In[ ]:


stats = pd.read_csv('C:Downloads\\DemographicData.csv')
stats


# In[ ]:


print(len(stats))


# In[ ]:


print(stats.shape)


# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[ ]:


plt.plot([1,2,3,4,5])
plt.show()


# In[ ]:


plt.plot([1,2,3,4,5],[6,7,8,9,0])
plt.show()


# In[ ]:


x=[10,20,30,40]
y=[50,60,70,80]
plt.plot(x,y)


# In[ ]:


x=[10,20,30,40]
y=[50,60,70,80]
plt.plot(x,y)
plt.title('First graph')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.show()


# In[ ]:


x=np.arange(0,16,2)
plt.plot(x,[x1**2 for x1 in x])


# In[ ]:


x=np.arange(0,16,2)
plt.plot(x,[x1 for x1 in x])
plt.plot(x,[x1**2 for x1 in x])
plt.plot(x,[x1**3 for x1 in x])


# In[ ]:


x=np.arange(0,16,2)
plt.plot(x,[x1 for x1 in x])
plt.plot(x,[x1**2 for x1 in x])
plt.plot(x,[x1**3 for x1 in x])
plt.axis([0,10,0,100])
plt.show()


# In[ ]:


x=np.arange(0,16,2)
plt.plot(x,[x1 for x1 in x])
plt.plot(x,[x1**2 for x1 in x])
plt.plot(x,[x1**3 for x1 in x])
plt.xlim([0,10])
plt.show()


# In[ ]:


from matplotlib import style
import matplotlib.style
style.use('ggplot')
x=[1,2,3,4,5]
y=[6,7,8,9,0]
a=[10,2,3,4,5]
b=[60,7,8,9,0]
plt.plot(a,b)
plt.plot(x,y)


# In[ ]:


a=[10,2,3,4,5]
b=[60,7,8,9,0]
plt.plot(a,b)


# In[ ]:


x=[10,12,14,16]
y=[20,30,15,25]
x1=[10.5,12.5,14.5,16.5]
y1=[20.5,30.5,15.5,25.5]
a=[11,13,15,17]
b=[21,31,16,26]
plt.plot(x,y,'orange',label='Line 1',linewidth=5)

plt.plot(x1,y1,'white',label='Line 2',linewidth=4)
plt.plot(a,b,'green',label='Line 3',linewidth=3)
plt.legend()
plt.title('First graph')
plt.xlabel('x-axis')
plt.ylabel('y-label')
plt.grid(True,color='k')

plt.show()


# In[ ]:


x=['10','20','30','40','50']
y=[20,30,15,25,30]

plt.plot(x,y)


for i in range(len(x)):
        plt.text(i,y[i],y[i],ha='left',va='bottom')


# In[ ]:


x=['10','20','30','40','50']
y=[20,30,15,25,30]

plt.bar(x,y)
plt.plot(x,y,color='b')


# In[ ]:


plt.bar([1,2,3,4,5],[10,11,12,13,14],label='Example 1',color='green',ec='red')
plt.bar([2,3,4,5,6],[6,7,8,9,0],label='Example 2',color='lightgreen',ec='blue')
plt.title('Graph 2')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.legend()
plt.show()


# In[ ]:


age=[1,2,3,4,5,6,7,8,9]
age_group=[0,1,2,3,4,5,6,7,8]
plt.hist(age,age_group)
plt.plot(age,age_group)
plt.bar(age,age_group,color='blue')
plt.show()


# In[ ]:


age=[1,2,3,4,5,6,7,8,9]
age_group=[0,1,2,3,4,5,6,7,8]
plt.scatter(age,age_group,color='orange')


# In[ ]:


age=[1,2,3,4,5,6,7,8,9]
age_group=[0,1,2,3,4,5,6,7,8]
plt.hist(age,age_group,histtype='step',color='k')
plt.scatter(age,age_group,color='red')


# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[ ]:


from matplotlib.pylab import rcParams
rcParams['figure.figsize']=15,6


# In[ ]:


days=[1,2,3,4,5]
sleep=[7,8,9,10,11]
eat=[2,3,4,5,6,7]
work=[8,9,9,7,1]
play=[7,4,1,1,1]
plt.stackplot(days,sleep,eat,work,play)
plt.show()


# In[ ]:


hrs=[100,2,7,3]
plt.pie(hrs)
plt.show()


# In[ ]:


hours=[5,8,3,4]
activity=['Sleep','Work','Eat','Play']
col=['g','r','c','y']
plt.pie(hours,labels=activity,colors=col,startangle=90,shadow=True,explode=(0,0,0,0.1),autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()


# In[ ]:





# In[10]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# In[11]:


os.getcwd()


# In[15]:


os.chdir ('C:\\Users\\Sagar\\Downloads\\')
os.getcwd()


# In[24]:


df1=pd.read_csv('Data Preprocessing Example.xlsx - Data.csv')
display (df1)


# In[21]:


x=df1.iloc[:,:-1],values
print(x)


# In[ ]:


y = df1.iloc[:,3].values
print (y)


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer = imputer.fit(x[:,0:1])
x [:,0:1]= imputer.transform(x[:,0:1])
print (x)


# In[ ]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=40)
imputer = imputer.fit(x[:,1:2])
x [:,1:2]= imputer.transform(x[:,1:2])
print (x)


# In[ ]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
print (x)


# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


# In[ ]:


os.getcwd()


# In[ ]:


os.chdir('C:\\Users\\Sagar\\Downloads')


# In[ ]:


os.getcwd()


# In[ ]:


df1=pd.read_csv('Salary_Data.csv')
df1


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 
import os


# In[ ]:


os.getcwd()


# In[ ]:





# In[ ]:




