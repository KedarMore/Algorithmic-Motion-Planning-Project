from scipy.interpolate import lagrange
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sympy.solvers import nsolve
from sympy import Symbol
import matplotlib as mpl


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

xlimits=(-2,12)
ylimits=(-5,5)
start=(0,0)
goal=(10,0)
obstacles=[[(3.5,4.5,4.5,3.5),(0.5,0.5,1.5,1.5)],
           [(6.5,7.5,7.5,6.5),(-1.5,-1.5,-0.5,-0.5)]]

# xlimits=(-2.,15.)
# ylimits=(-2.,15.)
# start=[0,0]
# goal=[10,10]
# obstacles=[[(1,2,2,1),(1,1,5,5)],
#            [(3,4,4,3),(4,4,12,12)],
#            [(3,12,12,3),(12,12,13,13)],
#            [(12,13,13,12),(5,5,13,13)],
#            [(6,12,12,6),(5,5,6,6)]]

# xlimits = (-6, 36)
# ylimits = (-6, 6)
# obstacles = [[(-6, 25, 25, -6), (-6, -6, -5, -5)],
#              [(-6, 30, 30, -6), (5, 5, 6, 6)],
#              [(-6, -5, -5, -6), (-5, -5, 5, 5)],
#              [(4, 5, 5, 4), (-5, -5, 1, 1)],
#              [(9, 10, 10, 9), (0, 0, 5, 5)],
#              [(14, 15, 15, 14), (-5, -5, 1, 1)],
#              [(19, 20, 20, 19), (0, 0, 5, 5)],
#              [(24, 25, 25, 24), (-5, -5, 1, 1)],
#              [(29, 30, 30, 29), (0, 0, 5, 5)]]
# start = [0, 0]
# goal = [35, 0]

pathi = []
pathj = []
with open('IFS.txt', 'r') as f:
    for line in f:
        for ele in range(len(line)):
            if line[ele] == '\t':
                br = ele
                break
        pathi.append(float(line[0:br]))
        pathj.append(float(line[br+1:-2]))

finalpath = list(np.transpose(np.vstack((pathi, pathj))))

f = []

for i in range(len(finalpath)):
    f.append(list(finalpath[i]))

finalpath=f

newfinalpath = []

newfinalpath.append(finalpath[0])

while str(newfinalpath[-1]) != str(finalpath[-1]):

    print(newfinalpath[-1])
    indx = finalpath.index(newfinalpath[-1])

    for i in range(indx, len(finalpath)):
        if i == len(finalpath)-1:
            newfinalpath.append(finalpath[-1])
            break
        if through_obstacle((finalpath[indx][0], finalpath[indx][1], finalpath[i][0], finalpath[i][1]), obstacles) == 1:
            newfinalpath.append(finalpath[i-1])
            break

newfinalpath = np.transpose(newfinalpath)

fig = plt.figure() 
  
ax = fig.add_subplot(111) 

ax.plot(*newfinalpath,color='orange')

for obs in obstacles:
    ax.fill(*obs, 'k', alpha=1)
plt.xlim(*xlimits)
plt.ylim(*ylimits)

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

ax.plot(totalx,totaly,color='blue')
# totalx.append(x)
# totaly.append(y)

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

print('x=',totalx)
print('theta=',theta)

plt.plot(start[0],start[1], 'o',color='red')
plt.plot(goal[0],goal[1], 'o',color='green')

plt.legend(["Eliminate Redundant Nodes","Smooth Curve","Start","Goal"])

plt.show()
