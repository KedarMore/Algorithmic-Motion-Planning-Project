import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sympy.solvers import nsolve
from sympy import Symbol
import matplotlib as mpl
import math

def inside_obstacle(point, obstacle):
    """
    returns 1 if the point is inside any obstacles
    0 otherwise
    """
    for obs in obstacle:
        if point[0] > obs[0][0] and point[0] < obs[0][2] and point[1] > obs[1][0] and point[1] < obs[1][2]:
            return 1
    return 0


def through_obstacle(line, obstacles):
    """
    returns 1 if the line goes through any obstacles
    0 otherwise
    """
    noofpoints = 100
    for i in range(noofpoints):
        if inside_obstacle((line[0]+(i*(line[2]-line[0])/noofpoints), line[1]+(i*(line[3]-line[1])/noofpoints)), obstacles) == 1:
            return 1
    return 0

def assign_values(qgoal,xlimits,ylimits):
    """
    assigns initial values to the grid
    """
    grid=np.zeros((len(xspace),len(yspace)))
    for i in range(len(xspace)):
        for j in range(len(yspace)):
            if xspace[i]==qgoal[0] and yspace[j]==qgoal[1]:
                grid[i][j]=2
                pass
            if inside_obstacle((xspace[i],yspace[j]),obstacles)==1:
                grid[i][j]=1
                pass
            pass
        pass
    return grid

def update(grid,number):
    """
    updates the grid to number+1
    """
    for i in range(len(xspace)):
        for j in range(len(yspace)):
            if grid[i][j]==number:
                if xspace[i]==qstart[0] and yspace[j]==qstart[1]:
                    return grid,number
                else:
                    if grid[(i+1)%len(xspace)][j]==0 and i+1<len(xspace):
                        grid[i+1][j]=number+1
                    if grid[max((i-1),0)][j]==0 and i-1>0:
                        grid[i-1][j]=number+1
                    if grid[i][(j+1)%len(yspace)]==0 and j+1<len(yspace):
                        grid[i][j+1]=number+1
                    if grid[i][max((j-1),0)]==0 and j-1>0:
                        grid[i][j-1]=number+1
                    
                    if grid[(i+1) % len(xspace)][(j+1) % len(yspace)] == 0 and i+1 < len(xspace) and j+1<len(yspace):
                        grid[i+1][j+1]=number+1
                    if grid[max((i-1), 0)][(j+1) % len(yspace)] == 0 and i-1 > 0 and j+1 < len(yspace):
                        grid[i-1][j+1]=number+1
                    if grid[max((i-1), 0)][max((j-1), 0)] == 0 and i-1>0 and j-1 > 0:
                        grid[i-1][j-1]=number+1
                    if grid[(i+1) % len(xspace)][max((j-1), 0)] == 0 and i+1 < len(xspace) and j-1 > 0:
                        grid[i+1][j-1] = number+1
                    pass
                pass
            pass
        pass
    return grid

def inside_obstacle(point,obstacle):
    """
    returns 1 if the point is inside any obstacles
    0 otherwise
    """
    for obs in obstacle:
        if point[0]>=obs[0][0] and point[0]<=obs[0][2] and point[1]>=obs[1][0] and point[1]<=obs[1][2]:
            return 1
    return 0

def find_path(grid,number):
    """
    plots a path from qgoal to qstart
    """
    pathcolour=number+10
    i=np.where(xspace==qstart[0])
    i=i[0][0]
    j=np.where(yspace==qstart[1])
    j=j[0][0]
    grid[(i)][j]=pathcolour

    igoal=np.where(xspace==qgoal[0])
    jgoal=np.where(yspace==qgoal[1])

    pathi=[]
    pathj=[]
    # pathi.append(xspace[i]+gridsize/2)
    # pathj.append(yspace[j]-gridsize/2)
    pathi.append(xspace[i])
    pathj.append(yspace[j])

    distance=0
    while number>2:

        distancetogoal=float('inf')

        if grid[(i+1) % len(xspace)][(j+1) % len(yspace)] == number-1:
            if np.sqrt((igoal-(i+1) % len(xspace))**2+(jgoal-(j+1) % len(yspace))**2) <= distancetogoal:
                inew = i+1
                jnew = j+1
                distancetogoal = np.sqrt((igoal-(i+1) % len(xspace))**2+(jgoal-(j+1) % len(yspace))**2)
    
        if grid[max((i-1), 0)][(j+1) % len(yspace)] == number-1:
            if np.sqrt((igoal-max((i-1), 0))**2+(jgoal-(j+1) % len(yspace))**2) <= distancetogoal:
                inew = i-1
                jnew = j+1
                distancetogoal = np.sqrt((igoal-max((i-1), 0))**2+(jgoal-(j+1) % len(yspace))**2)
    
        if grid[max((i-1),0)][max((j-1),0)] == number-1:
            if np.sqrt((igoal-max((i-1), 0))**2+(jgoal-max((j-1),0))**2) <= distancetogoal:
                inew = i-1
                jnew = j-1
                distancetogoal = np.sqrt((igoal-max((i-1), 0))**2+(jgoal-max((j-1),0))**2)
    
        if grid[(i+1)%len(xspace)][max((j-1),0)] == number-1:
            if np.sqrt((igoal-(i+1)%len(xspace))**2+(jgoal-max((j-1),0))**2) <= distancetogoal:
                inew = i+1
                jnew = j-1
                distancetogoal = np.sqrt((igoal-(i+1) % len(xspace))**2+(jgoal-max((j-1), 0))**2)

        #straight

        if grid[(i+1) % len(xspace)][j] == number-1:
            if np.sqrt((igoal-(i+1)%len(xspace))**2+(jgoal-(j))**2) <= distancetogoal:
                inew=i+1
                jnew=j
                distancetogoal = np.sqrt((igoal-(i+1) % len(xspace))**2+(jgoal-(j))**2)

        if grid[max((i-1),0)][j]==number-1:
            if np.sqrt((igoal-max((i-1), 0))**2+(jgoal-(j))**2) <= distancetogoal:
                inew=i-1
                jnew=j
                distancetogoal = np.sqrt((igoal-max((i-1), 0))**2+(jgoal-(j))**2)

        if grid[(i)][(j+1)%len(yspace)]==number-1:
            if np.sqrt((igoal-(i))**2+(jgoal-(j+1)%len(yspace))**2) <= distancetogoal:
                inew=i
                jnew=j+1
                distancetogoal = np.sqrt((igoal-(i))**2+(jgoal-(j+1) % len(yspace))**2)

        if grid[(i)][max((j-1),0)]==number-1:
            if np.sqrt((igoal-(i))**2+(jgoal-max((j-1),0))**2) <= distancetogoal:
                inew=i
                jnew=j-1
                distancetogoal = np.sqrt((igoal-(i))**2+(jgoal-max((j-1), 0))**2)

        if i!=inew and j!=jnew:
            distance=distance+gridsize*np.sqrt(2)
        else:
            distance = distance + gridsize
        
        number=number-1

        i=inew
        j=jnew

        # pathi.append(xspace[i]+gridsize/2)
        # pathj.append(yspace[j]-gridsize/2)
        pathi.append(xspace[i])
        pathj.append(yspace[j])
        
        pass
    return distance,pathi,pathj

def final(qgoal,xlimits,ylimits):
    grid=assign_values(qgoal,xlimits,ylimits)
    # fig,ax=plt.subplots()
    number=2
    while True:
        newgrid=update(grid,number)
        if len(newgrid)==2:
            grid,number=newgrid
            # print("found qstart on step number",number)
            break
        else:
            grid=newgrid
            number=number+1
            pass
    dist,pathi,pathj=find_path(grid,number)
    return dist, pathi, pathj

def smooth(pathi,pathj):
    
    finalpath = list(np.transpose(np.vstack((pathi, pathj))))

    f = []

    for i in range(len(finalpath)):
        f.append(list(finalpath[i]))

    finalpath=f

    newfinalpath = []

    newfinalpath.append(finalpath[0])

    while str(newfinalpath[-1]) != str(finalpath[-1]):

        # print(newfinalpath[-1])
        indx = finalpath.index(newfinalpath[-1])

        for i in range(indx, len(finalpath)):
            if i == len(finalpath)-1:
                newfinalpath.append(finalpath[-1])
                break
            if through_obstacle((finalpath[indx][0], finalpath[indx][1], finalpath[i][0], finalpath[i][1]), obstacles) == 1:
                newfinalpath.append(finalpath[i-1])
                break

    newfinalpath = np.transpose(newfinalpath)

    # fig = plt.figure() 
    
    # ax = fig.add_subplot(111) 

    # ax.plot(*newfinalpath,color='orange')

    # for obs in obstacles:
    #     ax.fill(*obs, 'k', alpha=1)
    # plt.xlim(*xlimits)
    # plt.ylim(*ylimits)

    pathi=newfinalpath[0]
    pathj=newfinalpath[1]

    a=Symbol('a')
    b=Symbol('b')
    c=Symbol('c')
    d=Symbol('d')
    e=Symbol('e')

    totalx=[]
    totaly=[]

    x=np.linspace(pathi[0],(pathi[0]+pathi[1])/2,20)
    y=np.linspace(pathj[0],(pathj[0]+pathj[1])/2,20)

    for points in range(len(x)):
        totalx.append(x[points])
        totaly.append(y[points])

    for i in range(int(len(pathi)-2)):
        f1=a*((pathi[i]+pathi[i+1])/2)**4+b*((pathi[i]+pathi[i+1])/2)**3+c*((pathi[i]+pathi[i+1])/2)**2+d*((pathi[i]+pathi[i+1])/2)**1+e-(pathj[i]+pathj[i+1])/2
        f2=a*((pathi[i+1]))**4+b*((pathi[i+1]))**3+c*((pathi[i+1]))**2+d*((pathi[i+1]))**1+e-(pathj[i+1])
        f3=a*((pathi[i+1]+pathi[i+2])/2)**4+b*((pathi[i+1]+pathi[i+2])/2)**3+c*((pathi[i+1]+pathi[i+2])/2)**2+d*((pathi[i+1]+pathi[i+2])/2)**1+e-(pathj[i+1]+pathj[i+2])/2

        f4=4*a*((pathi[i]+pathi[i+1])/2)**3+3*b*((pathi[i]+pathi[i+1])/2)**2+2*c*((pathi[i]+pathi[i+1])/2)**1+d-((pathj[i+1]-pathj[i])/(pathi[i+1]-pathi[i]))
        f5=4*a*((pathi[i+1]+pathi[i+2])/2)**3+3*b*((pathi[i+1]+pathi[i+2])/2)**2+2*c*((pathi[i+1]+pathi[i+2])/2)**1+d-((pathj[i+2]-pathj[i+1])/(pathi[i+2]-pathi[i+1]))

        variables=nsolve((f1,f2,f3,f4,f5),(a,b,c,d,e),(0,0,0,0,0))

        x=list(np.linspace((pathi[i]+pathi[i+1])/2,(pathi[i+1]+pathi[i+2])/2,20))

        for points in range(len(x)):
            totalx.append(x[points])
            totaly.append(variables[0]*x[points]**4+variables[1]*x[points]**3+variables[2]*x[points]**2+variables[3]*x[points]**1+variables[4])

    x=np.linspace((pathi[-1]+pathi[-2])/2,(pathi[-1]),20)
    y=np.linspace((pathj[-1]+pathj[-2])/2,(pathj[-1]),20)

    for points in range(len(x)):
        totalx.append(x[points])
        totaly.append(y[points])

    # ax.plot(totalx,totaly,color='blue')
    
    length=0
    for points in range(len(totalx)-1):
        length=length+math.sqrt((totalx[points]-totalx[points+1])**2+(totaly[points]-totaly[points+1])**2)
        # length=length+1

    # plt.plot([(pathi[-1]+pathi[-2])/2,pathi[-1]],[(pathj[-1]+pathj[-2])/2,pathj[-1]],color='blue')
    totalxdash=np.gradient(totalx)
    totalydash=np.gradient(totaly)
    totalydashdash=np.gradient(totalydash)

    R=[]
    for radius in range(len(totaly)):
        R.append(1/((totalydashdash[radius])*(1+(totalydash[radius])**2)**1.5))
        # print(R)

    # print('R=',R)

    l=1.5
    theta=[]
    for radii in R:
        theta.append(round(np.degrees(np.arctan(l/radii)),4))

    # print('x=',totalx)
    # print('theta=',theta)

    # plt.plot(qstart[0],qstart[1], 'o',color='red')
    # plt.plot(qgoal[0],qgoal[1], 'o',color='green')

    # plt.legend(["Eliminate Redundant Nodes","Smooth Curve","Start","Goal"])

    # plt.show()

    return length


if __name__ == "__main__":
    # xlimits=(-2,12)
    # ylimits=(-5,5)
    # qstart=(0,0)
    # qgoal=(10,0)
    # obstacles=[[(3.5,4.5,4.5,3.5),(0.5,0.5,1.5,1.5)],
    #            [(6.5,7.5,7.5,6.5),(-1.5,-1.5,-0.5,-0.5)]]

    # xlimits=(-2.,15.)
    # ylimits=(-2.,15.)
    # qstart=(0.00,0.00)
    # qgoal=(10.00,10.00)
    # obstacles=[[(1,2,2,1),(1,1,5,5)],
    #            [(3,4,4,3),(4,4,12,12)],
    #            [(3,12,12,3),(12,12,13,13)],
    #            [(12,13,13,12),(5,5,13,13)],
    #            [(6,12,12,6),(5,5,6,6)]]

    # xlimits=(-10,40)
    # ylimits=(-8,8)
    # qstart=(0,0)
    # qgoal=(35,0)
    # obstacles=[[(-6,25,25,-6),(-6,-6,-5,-5)],
    #            [(-6,30,30,-6),(5,5,6,6)],
    #            [(-6,-5,-5,-6),(-5,-5,5,5)],
    #            [(4,5,5,4),(-5,-5,1,1)],
    #            [(9,10,10,9),(0,0,5,5)],
    #            [(14,15,15,14),(-5,-5,1,1)],
    #            [(19,20,20,19),(0,0,5,5)],
    #            [(24,25,25,24),(-5,-5,1,1)],
    #            [(29,30,30,29),(0,0,5,5)]]

    gridsize=0.5

    # xspace=np.arange(xlimits[0],xlimits[1],gridsize)
    # xspace=np.round(xspace,2)
    # yspace=np.arange(ylimits[1],ylimits[0],-gridsize)
    # yspace=np.round(yspace,2)

    Ltotal=[]
    Ttotal=[]

    for obs in range(3):
        if obs==0:
            xlimits=(-2,12)
            ylimits=(-5,5)
            qstart=(0,0)
            qgoal=(10,0)
            obstacles=[[(3.5,4.5,4.5,3.5),(0.5,0.5,1.5,1.5)],
                    [(6.5,7.5,7.5,6.5),(-1.5,-1.5,-0.5,-0.5)]]
            xspace=np.arange(xlimits[0],xlimits[1],gridsize)
            xspace=np.round(xspace,2)
            yspace=np.arange(ylimits[1],ylimits[0],-gridsize)
            yspace=np.round(yspace,2)
        elif obs==1:
            xlimits=(-2.,15.)
            ylimits=(-2.,15.)
            qstart=(0.00,0.00)
            qgoal=(10.00,10.00)
            obstacles=[[(1,2,2,1),(1,1,5,5)],
                       [(3,4,4,3),(4,4,12,12)],
                       [(3,12,12,3),(12,12,13,13)],
                       [(12,13,13,12),(5,5,13,13)],
                       [(6,12,12,6),(5,5,6,6)]]
            xspace=np.arange(xlimits[0],xlimits[1],gridsize)
            xspace=np.round(xspace,2)
            yspace=np.arange(ylimits[1],ylimits[0],-gridsize)
            yspace=np.round(yspace,2)
        elif obs==2:
            xlimits=(-10,40)
            ylimits=(-8,8)
            qstart=(0,0)
            qgoal=(35,0)
            obstacles=[[(-6,25,25,-6),(-6,-6,-5,-5)],
                       [(-6,30,30,-6),(5,5,6,6)],
                       [(-6,-5,-5,-6),(-5,-5,5,5)],
                       [(4,5,5,4),(-5,-5,1,1)],
                       [(9,10,10,9),(0,0,5,5)],
                       [(14,15,15,14),(-5,-5,1,1)],
                       [(19,20,20,19),(0,0,5,5)],
                       [(24,25,25,24),(-5,-5,1,1)],
                       [(29,30,30,29),(0,0,5,5)]]
            xspace=np.arange(xlimits[0],xlimits[1],gridsize)
            xspace=np.round(xspace,2)
            yspace=np.arange(ylimits[1],ylimits[0],-gridsize)
            yspace=np.round(yspace,2)

        benchmark=100
        L=[]
        T=[]
        for i in range(benchmark):
            initial=time.time()
            path=final(qgoal,xlimits,ylimits)
            length1=smooth(path[1],path[2])
            L.append(length1)
            T.append(time.time()-initial)
        pass
        Ltotal.append(L)
        Ttotal.append(T)
    # for i in len(Ltotal):
    plt.boxplot(Ttotal)
    # plt.title("Benchmark for Path Length")
    plt.title("Benchmark for Computation Time")
    plt.show()