import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd

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
    pathi.append(xspace[i]+gridsize/2)
    pathj.append(yspace[j]-gridsize/2)

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

        pathi.append(xspace[i]+gridsize/2)
        pathj.append(yspace[j]-gridsize/2)
        
        pass


    return distance,pathi,pathj


if __name__ == "__main__":

    initialtime=time.time()

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

    gridsize=0.5

    xspace=np.arange(xlimits[0],xlimits[1],gridsize)
    xspace=np.round(xspace,2)
    yspace=np.arange(ylimits[1],ylimits[0],-gridsize)
    yspace=np.round(yspace,2)
    grid=assign_values(qgoal,xlimits,ylimits)
    # fig,ax=plt.subplots()
    number=2
    while True:
        newgrid=update(grid,number)
        if len(newgrid)==2:
            grid,number=newgrid
            print("found qstart on step number",number)
            break
        else:
            grid=newgrid
            number=number+1
            pass
    dist,pathi,pathj=find_path(grid,number)
    with open('IFS.txt', 'w') as f:
        for item in range(len(pathi)):
            f.write("%f\t" % float(pathi[item]-gridsize/2))
            f.write("%f\n" % float(pathj[item]+gridsize/2))
    print("Total distace of the path is:",np.round(dist,2))
    print("time taken",time.time()-initialtime)
    plt.imshow(grid.T,extent=[xlimits[0],xlimits[1],ylimits[0],ylimits[1]])
    plt.plot(pathi,pathj,color='red')
    plt.show()
    pass
