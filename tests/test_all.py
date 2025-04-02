from tba import *
import pytest

def test_fermi_surface():
    x = System(filling=0.40)
    fs = x.plot_Fermi_surface_contour(isExtendedZone=False, dk=np.pi/4,isShow=False)
    # expected Fermi surface segments
    s0 = np.array([0.06410413, 0.25      , 0.42299941, 0.5       , 0.67299941, 0.75      ])
    s1 = np.array([0.75      , 0.67299941, 0.5       , 0.42299941, 0.25      , 0.06410413])
    assert np.allclose(fs[0], s0)
    assert np.allclose(fs[1], s1)
