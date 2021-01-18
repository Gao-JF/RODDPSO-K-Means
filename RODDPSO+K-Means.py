# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 14:02:31 2021

@author: Gaojunfeng1020
"""

import sys
import math


import numpy as np
import matplotlib.pyplot as plt

data = """
1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459"""
 
#数据处理 dataset是30个样本（密度，含糖量）的列表
a = data.split(',')
dataset = [np.array([float(a[i]), float(a[i+1])]) for i in range(1, len(a)-1, 3)]


def getEuclidean(vec1, vec2):
    dist=vec1 - vec2
    dist=dist**2
    dist=dist[0]+dist[1]
    dist=dist**1/2
    return dist

def k_means(dataset, k, vectors):  
    fitness_list=[]
    pop_C=[]
    for vec in vectors:
        # 初始化簇
        C = []
        for i in range(K):
            C.append([vectors[i]])
        #初始化标签
        labels = []
        for i in range(len(dataset)):
            labels.append(-1)   
        for labelIndex in range(len(dataset)):
            item=dataset[labelIndex]
            classIndex = -1
            minDist = 1e6
            for i in range(vec.shape[1]):
                point=vec[:,i]
                dist = getEuclidean(item, point)
                if(dist < minDist):
                    classIndex = i
                    minDist = dist
            C[classIndex].append(item)
            labels[labelIndex] = classIndex
        for i in range(len(C)):
            C[i]=C[i][1:]
        fitness=[]
        total_d=0
        for i in range(len(C)):
            if len(C[i])==0:
                continue
            cluster=C[i]
            central=vec[:,i]
            clusterHeart = []
            dist=0
            for clu in cluster:
                dist+=getEuclidean(clu,central)
            total_d=dist/len(cluster)
            fitness.append(total_d)
            
        fitness=np.array(fitness)
        fitness=np.sum(fitness)/k
        fitness_list.append(fitness)
    return np.array(fitness_list)


    


#参数初始化
c1i=2.5
c1f=0.5
c2i=0.5
c2f=2.5

ml=0
mg=0

W_MAX=0.9
W_MIN=0.4
w=W_MAX

maxgen = 200   # 进化次数  
sizepop = 20   # 种群规模

# 粒子速度和位置的范围
Vmax =  0.02
Vmin = 0
popmax =  np.max(dataset)
popmin = np.min(dataset)

# 初始化聚类数
K=3

# 产生初始粒子和速度
pop=np.zeros([sizepop,2,K])
pop = np.random.uniform(popmin,popmax,size=[sizepop,2,K])
for i in range(sizepop):
    pop[i]=np.array([dataset[np.random.randint(0,len(dataset))],dataset[np.random.randint(0,len(dataset))],dataset[np.random.randint(0,len(dataset))]]).T
v = np.random.uniform(Vmin,Vmax,size=[sizepop,2,K])

#计算初始时每个粒子到其他粒子的平均距离
def dist(pop):
    d=np.zeros(sizepop)
    for pos in range(sizepop):
        d_sum=0
        for j in range(sizepop):
            if pos==j:
                continue
            temp=(pop[pos]-pop[j])**2
            dj_sum=0
            for n1 in range(temp.shape[0]):
                for n2 in range(temp.shape[1]):
                    dj_sum=dj_sum+temp[n1][n2]
            d_sum+=dj_sum**1/2
        d[pos]=d_sum/sizepop
    return d

        

d=dist(pop)                    # 初始时每个粒子到其他粒子的平均距离
#fitness = k_means(pop)             # 计算适应度
fitness = k_means(dataset,sizepop,pop) 
i = np.argmin(fitness)      # 找最好的个体
Ef=(d[i]-np.min(d))/(np.max(d)-np.min(d)) #计算进化因子
gbest = pop                    # 记录个体最优位置
gbest_history=[gbest]         # 记录个体最优位置历史值
zbest = pop[i]              # 记录群体最优位置
zbest_history=[zbest]         # 记录群体最优位置历史值
fitnessgbest = fitness        # 个体最佳适应度值
fitnesszbest = fitness[i]      # 全局最佳适应度值
#根据Ef寻找进化状态
if 0<=Ef<0.25:
    epsilonk=1
if 0.25<=Ef<0.5:
    epsilonk=2
if 0.5<=Ef<0.75:
    epsilonk=3
if 0.75<=Ef<=1:
    epsilonk=4
N=200
alpha=np.random.randint(2,size=N)
evo_state={1:(0,0),2:(0.01,0),3:(0,0.01),4:(0.01,0.01)}
ml,mg=evo_state[epsilonk]

# 迭代寻优
t = 0
record_RODDPSO = np.zeros(maxgen)
while t < maxgen:
    
    #惯性权重更新
    w=W_MAX-((W_MAX-W_MIN)*t/maxgen)
    
    #加速度系数更新
    c1=((c1f-c1i)*(maxgen-t)/maxgen)+c1i
    c2=((c2f-c2i)*(maxgen-t)/maxgen)+c2i
    c3=c1
    c4=c2
    
    #计算每个粒子的平均距离
    d=dist(pop)
    
    #更新Ef
    Ef=(d[i]-np.min(d))/(np.max(d)-np.min(d))
    
    #根据Ef寻找进化状态
    if 0<=Ef<0.25:
        epsilonk=1
    if 0.25<=Ef<0.5:
        epsilonk=2
    if 0.5<=Ef<0.75:
        epsilonk=3
    if 0.75<=Ef<=1:
        epsilonk=4
    
    ml,mg=evo_state[epsilonk]
    
    # 速度更新
    
    mcr3=0
    mcr4=0
    for tao in range(N):
        if t>=tao:
            mcr3+=alpha[tao]*(gbest_history[t-tao]-pop)
            mcr4+=alpha[tao]*(zbest_history[t-tao]-pop)
        else:
            mcr3+=alpha[tao]*(gbest_history[t]-pop)
            mcr4+=alpha[tao]*(zbest_history[t]-pop)
        
    v = w * v + c1 * np.random.random() * (gbest - pop) + c2 * np.random.random() * (zbest - pop)\
        +ml*c3* np.random.random()*mcr3+mg*c4*np.random.random()*mcr4
    v[v > Vmax] = Vmax     # 限制速度
    v[v < Vmin] = Vmin
    
    # 位置更新
    pop = pop + v;
    pop[pop > popmax] = popmax/2  # 限制位置
    pop[pop < popmin] = popmin/2
    
    '''
    # 自适应变异
    p = np.random.random()             # 随机生成一个0~1内的数
    if p > 0.8:                          # 如果这个数落在变异概率区间内，则进行变异处理
        k = np.random.randint(0,2)     # 在[0,2)之间随机选一个整数
        pop[:,k] = np.random.random()  # 在选定的位置进行变异 
    '''

    # 计算适应度值
    fitness = k_means(dataset,sizepop,pop)
    
    # 个体最优位置更新
    index = fitness < fitnessgbest
    fitnessgbest[index] = fitness[index]
    gbest[index] = pop[index]

    # 群体最优更新
    i = np.argmin(fitness)
    if fitness[i] < fitnesszbest:
        zbest = pop[i]
        fitnesszbest = fitness[i]
    
    #记录历史最优状态
    gbest_history.append(gbest)
    zbest_history.append(zbest)
    record_RODDPSO[t] = fitnesszbest # 记录群体最优位置的变化   
    
    t = t + 1
# 结果分析
print(zbest)
plt.figure()
plt.plot(record_RODDPSO,label='RODDPSO')
plt.xlabel('generation')  
plt.ylabel('fitness')  
plt.title('fitness curve(K-Means)')  
plt.legend()
plt.show()


C = []
for i in range(K):
    C.append([zbest[:,i]])
#初始化标签
labels = []
for i in range(len(dataset)):
    labels.append(-1)   
for labelIndex in range(len(dataset)):
    item=dataset[labelIndex]
    classIndex = -1
    minDist = 1e6
    for i in range(zbest.shape[1]):
        point=zbest[:,i]
        dist = getEuclidean(item, point)
        if(dist < minDist):
            classIndex = i
            minDist = dist
    C[classIndex].append(item)
    labels[labelIndex] = classIndex
    
plt.figure()
colors=['blue','red','green']
for i in range(len(C)):
    C[i]=C[i][1:]
for k in range(K):
    x=[]
    y=[]
    for i in range(len(C[k])):
        x.append(C[k][i][0])
        y.append(C[k][i][1])
    plt.scatter(x,y,color=colors[k])
plt.title('k-means with RODDPSO')
plt.xlabel('x')
plt.ylabel('y')
plt.show()