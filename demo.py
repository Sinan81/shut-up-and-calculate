import numpy as np

    def demo_sus_arcs(system):
        """
        inputs:
        system: a System type object
        """
        if type(system.crystal) != Tetra:
            print('This demo is only implemented for Tetra cyrstal')
            print('exiting...')
            exit()
        x,y = system.plot_Fermi_surface_contour(kmin=-pi,kmax=pi)
        q1x= []; q1y= []
        q2x= []; q2y= []
        q3x= []; q3y= []
        q4x= []; q4y= []
        # shift all x,y pairs the first quadrant
        xy = zip(x,y)
        for p in xy:
            px,py = p
            if px >= 0 and py >=0: # 1st quadrant
                q1x.append(px)
                q1y.append(py)
            if px < 0 and py >= 0: # 2nd quadrant
                q2x.append(px)
                q2y.append(py)
            if px < 0 and py < 0: # 3rd quadrant
                q3x.append(px)
                q3y.append(py)
            if px > 0 and py < 0: # 4th quadrant
                q4x.append(px)
                q4y.append(py)

        # plot data points only, line plot causes confusion
        # the goal is:
        pind=50
        askl=2.0 # scale factor
        for ashift in [2]:
            plt.plot(askl*np.array(q1x), askl*np.array(q1y),'o')
            plt.plot(askl*np.array(q2x)+2, askl*np.array(q2y),'o')
            plt.plot(askl*np.array(q3x)+2, askl*np.array(q3y)+2,'o')
            plt.plot(askl*np.array(q4x), askl*np.array(q4y)+2,'o')
            plt.xlim(0,2)
            plt.ylim(0,2)
            #plt.savefig('sus_arcs_anim_'+str(pind)+'.png')
            plt.show()
            pind=pind+1
        return None


