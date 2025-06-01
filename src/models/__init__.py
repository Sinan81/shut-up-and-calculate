from .model import Model
from .instance_cuprate_single_band import cuprate_single_band
from .instance_cuprate_three_band import cuprate_three_band
from .instance_cuprate_four_band_LCO import cuprate_four_band_LCO
from .instance_hexa_single_band import hexa_single_band
from .instance_tetra_single_band_ddw import tetra_single_band_ddw

# Optional: Make these available when importing the package directly
__all__ = [ 'Model',
            'cuprate_single_band',
            'cuprate_three_band',
            'cuprate_four_band_LCO',
            'hexa_single_band',
            'tetra_single_band_ddw'
            ]
