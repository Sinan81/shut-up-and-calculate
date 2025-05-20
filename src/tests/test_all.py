from tba import *
import pytest

# tests for single band tetra

def test_fermi_surface():
    x = System(filling=0.40)
    fs = x.plot_Fermi_surface_contour(isExtendedZone=False, dk=np.pi/4,isShow=False)
    # expected Fermi surface segments
    s0 = np.array([0.06410413, 0.25      , 0.42299941, 0.5       , 0.67299941, 0.75      ])
    s1 = np.array([0.75      , 0.67299941, 0.5       , 0.42299941, 0.25      , 0.06410413])
    assert np.allclose(fs[0], s0)
    assert np.allclose(fs[1], s1)


def test_band_along_a_cut():
    num=10
    x = System(filling=0.40)
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
    x = System(filling=0.40)
    chi_ref = 0.21320053290009247
    chi = x.chic.real_static(q)
    assert np.allclose(chi, chi_ref)


# multi band
def test_fermi_surface_tetra_three_band():
    x = System(model=cuprate_three_band,filling=2.40)
    fs = x.plot_Fermi_surface_contour(isExtendedZone=False, dk=np.pi/10,isShow=False)
    # expected Fermi surface segments
    cx_ref = np.array([0.06914328, 0.1       , 0.1349657 , 0.2       , 0.20846123,
            0.28383914, 0.3       , 0.36986394, 0.4       , 0.46873861,
            0.5       , 0.5826423 , 0.6       , 0.7       , 0.71545476,
            0.8       , 0.87557624, 0.9       ])
    cy_ref = np.array([0.9       , 0.87557624, 0.8       , 0.71545476, 0.7       ,
            0.6       , 0.5826423 , 0.5       , 0.46873861, 0.4       ,
            0.36986394, 0.3       , 0.28383914, 0.20846123, 0.2       ,
            0.1349657 , 0.1       , 0.06914328])
    assert np.allclose(fs[0], cx_ref)
    assert np.allclose(fs[1], cy_ref)

