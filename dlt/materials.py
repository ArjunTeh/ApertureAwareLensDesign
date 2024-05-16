from .constants import FLOAT_INFINITY
from .constants import DEFAULT_EXT_IOR

# copied from https://github.com/vccimaging/DiffDeflectometry/blob/master/diffmetrology/basics.py
MATERIAL_TABLE = { # [nD, Abbe number]
    "vacuum":     [1.,       FLOAT_INFINITY],
    "air":        [1.000293, FLOAT_INFINITY],
    "occulder":   [1.,       FLOAT_INFINITY],
    "f2":         [1.620,    36.37],
    "f15":        [1.60570,  37.831],
    "uvfs":       [1.458,    67.82],
    
    # https://shop.schott.com/advanced_optics/
    "bk10":       [1.49780,  66.954],
    "n-baf10":    [1.67003,  47.11],
    "n-baf52":    [ 1.60863, 46.60],
    "n-bk7":      [1.51680,  64.17],
    "n-sf1":      [1.71736,  29.62],
    "n-sf2":      [1.64769,  33.82],
    "n-sf4":      [1.75513,  27.38],
    "n-sf5":      [1.67271,  32.25],
    "n-sf6":      [1.80518,  25.36],
    "n-sf6ht":    [1.80518,  25.36],
    "n-sf8":      [1.68894,  31.31],
    "n-sf10":     [1.72828,  28.53],
    "n-sf11":     [1.78472,  25.68],
    "sf1":        [1.71736,  29.51],
    "sf2":        [1.64769,  33.85],
    "sf4":        [1.75520,  27.58],
    "sf5":        [1.67270,  32.21],
    "sf6":        [1.80518,  25.43],
    "sf18":       [1.72150,  29.245],

    # HIKARI.AGF
    "baf10":      [1.67,     47.05],

    # SUMITA.AGF
    "sk16":       [1.62040,  60.306],
    "sk1":        [1.61030,  56.712],
    "ssk4":       [1.61770,  55.116],

    # https://www.pgo-online.com/intl/B270.html
    "b270":       [1.52290,  58.50],
    
    # https://refractiveindex.info, nd at 589.3 [nm]
    "s-nph1":     [1.8078,   22.76], 
    "d-k59":      [1.5175,   63.50],
    
    "flint":      [1.6200,   36.37],
    "pmma":       [1.491756, 58.00],
    "polycarb":   [1.585470, 30.00],

    # OHARA.AGF
    "s-lah66":    [1.772499, 49.598],
    "s-fsl5":     [1.487490, 70.236252],
    "s-lal56":    [1.677898, 50.722184],
    "s-lal8":     [1.712995, 53.867058],
    "s-tih11":    [1.784723, 25.683446],
    "s-lal61":    [1.740999, 52.636502],
    "s-bal14":    [1.568832, 56.363897],
    "s-tih6":     [1.805181, 25.425363],
    "s-lal18":    [1.729157, 54.680013],
    "h-f1":       [1.6034, 38.011],
    "s-nbh8":     [1.7205, 34.708],
}


def get_nd(mat_name : str, glass_cat=None):
    if glass_cat is None:
        glass_cat = MATERIAL_TABLE

    mat_name = mat_name.upper()
    if mat_name in glass_cat:
        return True, glass_cat[mat_name][0]
    else:
        return False, None


def get_eta_by_wavelength(mat_name : str, glass_cat):
    mat_name = mat_name.upper()
    if mat_name in glass_cat:
        return True, glass_cat[mat_name][2]
    else:
        return False, lambda x : DEFAULT_EXT_IOR