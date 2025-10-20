from tba import *
import pytest
import pdb

# tests for single band tetra

def test_fermi_surface():
    x = CuprateSingleBand(filling=0.40)
    fs = x.plot_Fermi_surface_contour(isExtendedZone=False, dk=np.pi/4,isShow=False)
    # expected Fermi surface segments
    s0 = np.array([ 0.75,        0.67299941,  0.5,         0.42299941,  0.25,        0.06410413,
                    0.,         -0.06410413, -0.25,       -0.42299941, -0.5,        -0.67299941,
                    -0.75,       -0.81410413, -0.75,       -0.67299941, -0.5,        -0.42299941,
                    -0.25,       -0.06410413,  0.06410413,  0.25,        0.42299941,  0.5,
                    0.67299941,  0.75      ])

    s1 = np.array( [-0.06410413, -0.25,       -0.42299941, -0.5,        -0.67299941, -0.75,
             -0.81410413, -0.75, -0.67299941, -0.5,        -0.42299941, -0.25,
             -0.06410413,  0. ,         0.06410413,  0.25,        0.42299941,  0.5,
              0.67299941, 0.75, 0.75, 0.67299941,  0.5,         0.42299941,
              0.25, 0.06410413])
    #pdb.set_trace()
    assert np.allclose(fs[0], s0)
    assert np.allclose(fs[1], s1)


def test_band_along_a_cut():
    num=10
    x = CuprateSingleBand(filling=0.40)
    p1,p2 = x.crystal.sym_cuts[0]
    lkx = np.linspace(p1[0], p2[0], num=num)
    lky = np.linspace(p1[1], p2[1], num=num)
    veband = np.vectorize(x.Eband)  # vectorize
    Z = veband(lkx,lky)
    Z_ref = np.array([-4., -3.87938524, -3.53208889, -3., -2.34729636,
                        -1.65270364, -1., -0.46791111, -0.12061476,  0.])
    assert np.allclose(Z,Z_ref)


def test_susceptibility_charge_real_static():
    # very heavy computation so just pick a single point to calculate
    q = (pi/2,pi/2)
    x = CuprateSingleBand(filling=0.40)
    chi_ref = 0.21320053290009247
    chi = x.chic.real_static(q)
    assert np.allclose(chi, chi_ref)

def test_susceptibility_current_real_static():
    # very heavy computation so just pick a single point to calculate
    x = CuprateSingleBand()
    chi_ref = 0.1500234000455209
    out = x.chij.calc_vs_q(Nq=1)
    chi = np.real(out[0][0][0][0])
    assert np.allclose(chi, chi_ref)


def test_current_sus_factors():
    # very heavy computation so just pick a single point to calculate
    x = CuprateSingleBand()
    k = (0.1, 0.2)
    q = (0.3, 0.4)
    kpq =(k[0]+q[0],k[1]+q[1])
    mq = (-q[0], -q[1])
    assert x.h1a_right(k,q) == x.h1a(kpq,mq)
    assert x.h1b_right(k,q) == x.h1b(kpq,mq)
    assert x.h2a_right(k,q) == x.h2a(kpq,mq)
    assert x.h2b_right(k,q) == x.h2b(kpq,mq)
    assert x.h3a_right(k,q) == x.h3a(kpq,mq)
    assert x.h3b_right(k,q) == x.h3b(kpq,mq)
    assert x.h4a_right(k,q) == x.h4a(kpq,mq)
    assert x.h4b_right(k,q) == x.h4b(kpq,mq)


# multi band
def test_fermi_surface_tetra_three_band():
    x = CuprateThreeBand(filling=2.40)
    fs = x.plot_Fermi_surface_contour(isExtendedZone=False, dk=np.pi/4,isShow=False)
    # expected Fermi surface segments
    cx_ref = np.array([-0.01181665, -0.12964988, -0.25      , -0.36733857, -0.5       ,
           -0.66032169, -0.75      , -1.        ,  0.75      ,  0.66032169,
            0.5       ,  0.36733857,  0.25      ,  0.12964988,  0.01181665,
           -1.        , -0.75      , -0.66032169, -0.5       , -0.36733857,
           -0.25      , -0.12964988,  0.12964988,  0.25      ,  0.36733857,
            0.5       ,  0.66032169,  0.75      ])
    cy_ref = np.array([-1.        , -0.75      , -0.66032169, -0.5       , -0.36733857,
           -0.25      , -0.12964988, -0.01181665, -0.12964988, -0.25      ,
           -0.36733857, -0.5       , -0.66032169, -0.75      , -1.        ,
            0.01181665,  0.12964988,  0.25      ,  0.36733857,  0.5       ,
            0.66032169,  0.75      ,  0.75      ,  0.66032169,  0.5       ,
            0.36733857,  0.25      ,  0.12964988])
    assert np.allclose(fs[0], cx_ref)
    assert np.allclose(fs[1], cy_ref)

def test_fill_vs_energy_multi():
    x = CuprateThreeBand()
    fill_ref = np.array([2.38953066e-03, 5.76927926e-01, 1.94393953e+00, 2.00000000e+00,
            2.00000000e+00, 2.12335469e+00, 2.63501520e+00, 2.97435109e+00])
    energy,fill = x.filling_vs_energy(dE=1, isplot=False)
    assert np.allclose(fill,fill_ref)
