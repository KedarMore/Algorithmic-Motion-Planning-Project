import matplotlib.pyplot as plt
import numpy as np
import math
import time

# obstacles

# xlimits=(-2,12)
# ylimits=(-5,5)
# start=(0,0)
# goal=(10,0)
# obstacles=[[(3.5,4.5,4.5,3.5),(0.5,0.5,1.5,1.5)],
#            [(6.5,7.5,7.5,6.5),(-1.5,-1.5,-0.5,-0.5)]]

# xlimits=(-2.,15.)
# ylimits=(-2.,15.)
# start=(0.00,0.00)
# goal=(10.00,10.00)
# obstacles=[[(1,2,2,1),(1,1,5,5)],
#             [(3,4,4,3),(4,4,12,12)],
#             [(3,12,12,3),(12,12,13,13)],
#             [(12,13,13,12),(5,5,13,13)],
#             [(6,12,12,6),(5,5,6,6)]]

# xlimits=(-10,40)
# ylimits=(-8,8)
# start=(0,0)
# goal=(35,0)
# obstacles=[[(-6,25,25,-6),(-6,-6,-5,-5)],
#            [(-6,30,30,-6),(5,5,6,6)],
#            [(-6,-5,-5,-6),(-5,-5,5,5)],
#            [(4,5,5,4),(-5,-5,1,1)],
#            [(9,10,10,9),(0,0,5,5)],
#            [(14,15,15,14),(-5,-5,1,1)],
#            [(19,20,20,19),(0,0,5,5)],
#            [(24,25,25,24),(-5,-5,1,1)],
#            [(29,30,30,29),(0,0,5,5)]]

def dist(a,b):
    return np.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

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
    noofpoints = 20
    for i in range(noofpoints):
        if inside_obstacle((line[0]+(i*(line[2]-line[0])/noofpoints), line[1]+(i*(line[3]-line[1])/noofpoints)), obstacles) == 1:
            return 1
    return 0

class node:
    def __init__(self,point,parent):
        self.point=point
        self.parent=parent
        self.pathlength=0

def nearest_node(nodelist, point):
    maxdist=float('inf')
    for l in nodelist:
        distance=dist(point,l.point)
        if distance<maxdist:
            nn=l
            maxdist=distance
    return nn

def steer(node, point):
    distance=2
    if dist(node.point,point)<distance:
        return point
    else:
        angle=np.arctan2(point[1]-node.point[1],point[0]-node.point[0])
        return [node.point[0]+distance*np.cos(angle),node.point[1]+distance*np.sin(angle)]

def costtostart(point,nodelist):
    for n in nodelist:
        if point==n.point:
            start=n
    cost=0
    while start.parent is not None:
        cost=cost+dist(start.point,start.parent)
        for n in nodelist:
            if start.parent==n.point:
                start=n
    return cost

finaltime=[]
finallength=[]
finalcorrectvalue=[]

for obs in range(2,3):

    if obs==0:
        xlimits=(-2,12)
        ylimits=(-5,5)
        start=(0,0)
        goal=(10,0)
        obstacles=[[(3.5,4.5,4.5,3.5),(0.5,0.5,1.5,1.5)],
                   [(6.5,7.5,7.5,6.5),(-1.5,-1.5,-0.5,-0.5)]]

    if obs==1:
        xlimits=(-2.,15.)
        ylimits=(-2.,15.)
        start=(0.00,0.00)
        goal=(10.00,10.00)
        obstacles=[[(1,2,2,1),(1,1,5,5)],
                    [(3,4,4,3),(4,4,12,12)],
                    [(3,12,12,3),(12,12,13,13)],
                    [(12,13,13,12),(5,5,13,13)],
                    [(6,12,12,6),(5,5,6,6)]]
    if obs==2:
        xlimits=(-10,40)
        ylimits=(-8,8)
        start=(0,0)
        goal=(35,0)
        obstacles=[[(-6,25,25,-6),(-6,-6,-5,-5)],
                   [(-6,30,30,-6),(5,5,6,6)],
                   [(-6,-5,-5,-6),(-5,-5,5,5)],
                   [(4,5,5,4),(-5,-5,1,1)],
                   [(9,10,10,9),(0,0,5,5)],
                   [(14,15,15,14),(-5,-5,1,1)],
                   [(19,20,20,19),(0,0,5,5)],
                   [(24,25,25,24),(-5,-5,1,1)],
                   [(29,30,30,29),(0,0,5,5)]]

    fig, ax = plt.subplots()

    plt.xlim(*xlimits)
    plt.ylim(*ylimits)

    for obst in obstacles:
        ax.fill(obst[0],obst[1],color='k')

    length=[]
    timearray=[]
    correctValue=0
    for benchmark in range(100):
        
        initial=time.time()
        n=node(start,None)
        nodelist=[]
        nodelist.append(n)

        #initial

        v=3
        l=1
        steeringlimits=[-0.524,0.524]
        # steeringlimits=[-1.524,1.524]
        theta=0
        shorttime=0.01
        x=start[0]
        y=start[1]
        # correctValue=0

        for iteration in range(500):

            pathlength=0
            #sample points
            xrand=np.random.uniform(*xlimits,1)
            yrand=np.random.uniform(*ylimits,1)


            nearestnode=nearest_node(nodelist,(xrand,yrand))
            newpoint=steer(nearestnode,(xrand,yrand))

            if through_obstacle((*nearestnode.point,*newpoint),obstacles)==1:
                continue
            
            # newpoint

            mindist=float('inf')
            for i in range(100):
                steeringangle=np.random.uniform(*steeringlimits,1)
                thetafinal=(v/l)*math.tan(steeringangle)*shorttime*10
                # if abs(thetafinal-theta)>abs(-thetafinal-theta):
                #     thetafinal=-thetafinal
                xfinal=v*math.cos(theta)*shorttime*10
                yfinal=v*math.sin(theta)*shorttime*10
                if dist(newpoint,[xfinal,yfinal])<=mindist:
                    mindist=dist(newpoint,[xfinal,yfinal])
                    x=xfinal
                    y=yfinal
                    theta=thetafinal
                    beststeeringangle=steeringangle

            tempx=[]
            tempy=[]
            flag=0
            for i in range(1,101):
                thetafinal=(v/l)*math.tan(beststeeringangle)*shorttime*i
                # if abs(thetafinal-theta)>abs(-thetafinal-theta):
                #     thetafinal=-thetafinal
                xfinal=nearestnode.point[0]+v*math.cos(theta)*shorttime*i
                yfinal=nearestnode.point[1]+v*math.sin(theta)*shorttime*i
                tempx.append(xfinal)
                tempy.append(yfinal)
                if inside_obstacle((xfinal,yfinal),obstacles)==1 or xfinal>=xlimits[1] or xfinal<=xlimits[0] or yfinal>=ylimits[1] or yfinal<=ylimits[0]:
                    flag=1
                theta=thetafinal
            if flag==1:
                continue
            pathlength=pathlength+dist((tempx[0],tempy[0]),nearestnode.point)
            plt.plot((tempx[0],nearestnode.point[0]),(tempy[0],nearestnode.point[1]),'r')
            for i in range(len(tempx)-1):
                pathlength=pathlength+dist((tempx[i],tempy[i]),(tempx[i+1],tempy[i+1]))
                plt.plot((tempx[i],tempx[i+1]),(tempy[i],tempy[i+1]),'r')
            pathlength=pathlength+dist((tempx[-1],tempy[-1]),(xfinal,yfinal))
            plt.plot((tempx[-1],xfinal),(tempy[-1],yfinal),'r')
            n=node((xfinal,yfinal),nearestnode.point)
            n.pathlength=pathlength
            nodelist.append(n)
            flagin="out"
            if dist((xfinal,yfinal),goal)<=2:
                # print("found")
                flagin="in"
                break
        if flagin=="in":
            correctValue=correctValue+1
            currnode=n
            pathlength=0
            while currnode.point[0]!=start[0] and currnode.point[1]!=start[1]:
                # print(currnode.parent[0])
                pathlength=pathlength+currnode.pathlength
                for i in nodelist:
                    if currnode.parent[0]==i.point[0] and currnode.parent[1]==i.point[1]:
                        # print(i.point)
                        currnode=i
                        break
            timearray.append(time.time()-initial)
            length.append(pathlength)

    finaltime.append(timearray)
    finallength.append(length)
    finalcorrectvalue.append(correctValue)

plt.show()

plt.boxplot(finaltime)
plt.title("Benchmark for Computation Time")
plt.show()

plt.boxplot(finallength)
plt.title("Benchmark for Path Length")
plt.show()

print(finalcorrectvalue)
