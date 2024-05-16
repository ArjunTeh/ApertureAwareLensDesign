import numpy as np

import os
import warnings
import collections

from . import file_io
from . import materials
from . import constants


class ZemaxSurfaceType():
    STANDARD = 'STANDARD'
    EVENASPH = 'EVENASPH'
    STOP = 'STOP'
    CONIC = 'CONIC'
    SOURCE_PLANE = 'SOURCEPLANE'
    IMAGE_PLANE = 'IMAGEPLANE'

    @staticmethod
    def to_state(surf_type):
        surf_type = surf_type.upper()
        if surf_type == ZemaxSurfaceType.STANDARD:
            return ZemaxSurfaceType.STANDARD
        elif surf_type == ZemaxSurfaceType.EVENASPH:
            return ZemaxSurfaceType.EVENASPH


def load_zemax_generic(fname, material_override=False):
    surfaces, wavelengths, glass_cat_names = file_io.parse_zemax_file(fname)
    glass_cat = load_glass_catalogs(*glass_cat_names)

    surface_descriptions = [None] * len(surfaces)
    for i, s in enumerate(surfaces):
        surf_id = s[0]
        fields = s[1]

        if "GLAS" not in fields:
            surf_mat = "AIR"
            ior_fn = load_glass_ior_function("AIR")
            ior_nd = constants.DEFAULT_EXT_IOR
            ior_abbe = 0.0
        else:
            surf_mat = fields["GLAS"][0]
            if material_override or surf_mat not in glass_cat:
                warnings.warn(f"material {surf_mat} not found, approximating ior with Abbe number")
                ior_nd = fields['GLAS'][1][2]
                ior_abbe = fields['GLAS'][1][3]
                ior_fn = load_glass_ior_cauchy(ior_nd, ior_abbe)
                # warnings.warn(f"Material {surf_mat} not found in glass catalog, using default ior")
                # surf_mat = "AIR"
                # ior_fn = load_glass_ior_function("AIR")
                # ior_nd = constants.DEFAULT_EXT_IOR
                # ior_abbe = 0.0
            else:
                ior_nd = glass_cat[surf_mat][0]
                ior_abbe = glass_cat[surf_mat][1]
                ior_fn = glass_cat[surf_mat][2]

        if "PARM" in fields:
            conic_parameter = fields.get("CONI", 0.0)
            asphere_parameters = fields["PARM"]
        else:
            conic_parameter = 0.0
            asphere_parameters = [0.0]*8

        is_stop = "STOP" in fields

        if i == 0:
            surf_type = ZemaxSurfaceType.SOURCE_PLANE
        elif i == len(surfaces) - 1:
            surf_type = ZemaxSurfaceType.IMAGE_PLANE
        else:
            if is_stop:
                surf_type = ZemaxSurfaceType.STOP
            else:
                surf_type = ZemaxSurfaceType.to_state(fields['type'])

        # check to see if there are zoom configurations
        multiple_configs = False
        distance = fields['DISZ']
        if "THIC" in fields:
            multiple_configs = True
            dist_list = fields["THIC"] 
            distance = [distance]*len(dist_list)
            for config_idx, dist_val in dist_list:
                distance[config_idx-1] = dist_val

        semidiam = fields['DIAM']
        if "SDIA" in fields:
            multiple_configs = True
            diam_list = fields["SDIA"]
            semidiam = [semidiam]*len(diam_list)
            for config_idx, diam_val in diam_list:
                semidiam[config_idx-1] = diam_val


        surf_desc = dict(
            surf_id=surf_id,
            surf_type=surf_type,
            curvature=fields['CURV'],
            distance=distance,
            semidiam=semidiam,
            ior_fn=ior_fn,
            ior_nd=ior_nd,
            ior_abbe=ior_abbe,
            mat_name=surf_mat,
            conic_parameter=conic_parameter,
            asphere_parameters=asphere_parameters,
            is_stop=is_stop,
            multiple_configs=multiple_configs,
        )
        surface_descriptions[surf_id] = surf_desc

    return surface_descriptions


def desc_state_to_array(desc_state, wavelength=587.56, config_id=0):
    lens = np.zeros((len(desc_state)-2, 13))
    for i, s in enumerate(desc_state[1:-1]):

        lens[i, 0] = s['curvature']
        lens[i, 1] = s['distance'] if isinstance(s['distance'], float) else s['distance'][config_id]
        lens[i, 2] = s['ior_fn'](wavelength)
        lens[i, 3] = s['semidiam'] if isinstance(s['semidiam'], float) else s['semidiam'][config_id]
        lens[i, 4] = s['conic_parameter']
        lens[i, 5:13] = s['asphere_parameters']

    lens_type_list = [s['surf_type'] for s in desc_state[1:-1]]

    if config_id == -1:
        surf_ids = []
        distances = []
        for lens_info in desc_state:
            if not isinstance(lens_info['distance'], float):
                distances.append(lens_info['distance'])
                surf_ids.append(lens_info['surf_id']-1)
        distances = np.array(distances).T
        return lens, lens_type_list, (surf_ids, distances)
    
    return lens, lens_type_list


def array_to_desc_state(lens, old_desc_state):
    new_desc = old_desc_state.copy()
    for i, l in enumerate(lens):
        new_desc[i+1]['curvature'] = l[0]
        new_desc[i+1]['distance'] = l[1]
        new_desc[i+1]['semidiam'] = l[3]
        new_desc[i+1]['conic_parameter'] = l[4]
        new_desc[i+1]['asphere_parameters'] = l[5:13]
    return new_desc


def load_zemax_file(fname, non_standard=False, info_list=False):
    surfaces, wavelengths, glass_cat_names = file_io.parse_zemax_file(fname)

    glass_cat = load_glass_catalogs(*glass_cat_names)

    # first surface is the focal plane which is sometimes at infinity
    # last surface is the sensor which is placed at the origin?
    if non_standard:
        state = np.zeros((len(surfaces[1:-1]), 13))
    else:
        state = np.zeros((len(surfaces[1:-1]), 4))

    surf_type_list = []
    for i, s in enumerate(surfaces[1:-1]):
        fields = s[1]
        if s[1]['type'] not in ['STANDARD', 'EVENASPH']:
            raise ValueError("Only standard and even aspheres are supported at this time!")
        
        if "STOP" in fields:
            surf_type_list.append(ZemaxSurfaceType.STOP)
        else:
            surf_type_list.append(ZemaxSurfaceType.to_state(fields['type']))

        if "GLAS" not in fields:
            surf_ior = constants.DEFAULT_EXT_IOR
        else:
            # TODO need to add dispersion formulas after reading in nd and Vd
            # https://wiki.luxcorerender.org/Glass_Material_IOR_and_Dispersion#Conversion_from_Abbe_Number_to_Cauchy
            # discussion of the d, F, and C wavelengths
            # https://wiki.luxcorerender.org/Glass_Material_IOR_and_Dispersion#The_Abbe_number
            surf_mat = fields["GLAS"]
            in_material_list, nd = materials.get_nd(surf_mat[0], glass_cat)
            surf_ior = nd if in_material_list else surf_mat[1][2]

        state[i, 0] = fields["CURV"]
        state[i, 1] = fields["DISZ"]
        state[i, 2] = surf_ior
        state[i, 3] = fields["DIAM"]

        if non_standard and ("PARM" in fields):
            assert len(fields["PARM"]) == 8, "File does not have the correct number of parameters"
            if "CONI" in fields:
                state[i, 4] = fields["CONI"]
            state[i, 5:13] = fields["PARM"]

    if info_list:
        return state, surf_type_list
    return state


def export_zemax_state_to_file(zemax_state, out_file, ref_file):
    lens_info = []

    for surf in zemax_state:
        curv = f"{surf[0].item():.18E}"
        dist = f"{surf[1].item():.18E}"
        diam = f"{surf[3].item():.18E}"
        lens_info.append((curv, dist, diam))

    file_io.export_zemax_file(lens_info, out_file, ref_file)


def load_glass_catalogs(*catalog_names):
    glass_cat = dict()
    for cn in catalog_names:
        fname = os.path.join("data/zemax_files/glass_catalogs", cn.upper() + ".AGF")

        if not os.path.isfile(fname):
            print(cn.upper(), "catalog does not found")
            continue

        glass_raw_info = file_io.parse_zemax_glass_catalog(fname)

        for key, val in glass_raw_info.items():
            nd = float(val['nm'][2])
            abbe = float(val['nm'][3])
            wav_func = load_glass_ior_function(key, val)
            
            if key not in glass_cat:
                glass_cat[key] = (nd, abbe, wav_func)
            # else:
            #     warnings.warn(f"Multiple glass with the same name {key}, keeping the first one")

    return glass_cat


def load_glass_ior_function(glass_name, info=None):
        if info is not None:
            dispform = int(float(info['nm'][0]))
            cd = [float(c) for c in info['cd']]

        ## Code pulled from Zemax's glass catalog reader
        ## https://github.com/nzhagen/zemaxglass/blob/bb2486d0367dc97886627493096ad74dd6a32aa4/ZemaxGlass.py#L319C1-L364C1
        if glass_name.upper() == 'VACUUM':
            def ior(w):
                return 1.0

        if (glass_name.upper() == 'AIR') or (dispform == 0):
            ## use this for AIR and VACUUM
            def ior(w):
                return 1+0.05792105/(238.0185-w**-2)+0.00167917/(57.362-w**-2)

        elif (dispform == 1):
            def ior(w):
                formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + (cd[5] * w**-8)
                return np.sqrt(formula_rhs)

        elif (dispform == 2):  ## Sellmeier1
            def ior(w):
                formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + (cd[4] * w**2 / (w**2 - cd[5]))
                return np.sqrt(formula_rhs + 1.0)

        elif (dispform == 3):  ## Herzberger
            def ior(w):
                L = 1.0 / (w**2 - 0.028)
                indices = cd[0] + (cd[1] * L) + (cd[2] * L**2) + (cd[3] * w**2) + (cd[4] * w**4) + (cd[5] * w**6)

        elif (dispform == 4):  ## Sellmeier2
            def ior(w):
                formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - (cd[2])**2)) + (cd[3] / (w**2 - (cd[4])**2))
                return np.sqrt(formula_rhs + 1.0)

        elif (dispform == 5):  ## Conrady
            def ior(w):
                return cd[0] + (cd[1] / w) + (cd[2] / w**3.5)

        elif (dispform == 6):  ## Sellmeier3
            def ior(w):
                formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                                (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7]))
                return np.sqrt(formula_rhs + 1.0)

        elif (dispform == 7):  ## HandbookOfOptics1
            def ior(w):
                formula_rhs = cd[0] + (cd[1] / (w**2 - cd[2])) - (cd[3] * w**2)
                return np.sqrt(formula_rhs)

        elif (dispform == 8):  ## HandbookOfOptics2
            def ior(w):
                formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) - (cd[3] * w**2)
                return np.sqrt(formula_rhs)

        elif (dispform == 9):  ## Sellmeier4
            def ior(w):
                formula_rhs = cd[0] + (cd[1] * w**2 / (w**2 - cd[2])) + (cd[3] * w**2 / (w**2 - cd[4]))
                return np.sqrt(formula_rhs)

        elif (dispform == 10):  ## Extended1
            def ior(w):
                formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                              (cd[5] * w**-8) + (cd[6] * w**-10) + (cd[7] * w**-12)
                return np.sqrt(formula_rhs)

        elif (dispform == 11):  ## Sellmeier5
            def ior(w):
                formula_rhs = (cd[0] * w**2 / (w**2 - cd[1])) + (cd[2] * w**2 / (w**2 - cd[3])) + \
                              (cd[4] * w**2 / (w**2 - cd[5])) + (cd[6] * w**2 / (w**2 - cd[7])) + \
                              (cd[8] * w**2 / (w**2 - cd[9]))
                return np.sqrt(formula_rhs + 1.0)

        elif (dispform == 12):  ## Extended2
            def ior(w):
                formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**-2) + (cd[3] * w**-4) + (cd[4] * w**-6) + \
                                (cd[5] * w**-8) + (cd[6] * w**4) + (cd[7] * w**6)
                return np.sqrt(formula_rhs)

        elif (dispform == 13):  ## Extended3
            def ior(w):
                formula_rhs = cd[0] + (cd[1] * w**2) + (cd[2] * w**4)
                formula_rhs += (cd[3] * w**-2) + (cd[4] * w**-4) + (cd[5] * w**-6) + (cd[6] * w**-8) + (cd[7] * w**-10) + (cd[8] * w**-12)
                return np.sqrt(formula_rhs)

        else:
            raise ValueError('Dispersion formula #' + str(dispform) + ' (for glass=' + glass_name + ') is not a valid choice.')
        
        # need to convert nm to microns for the formula
        def ior_um(w):
            return ior(w * 1e-3)
        return ior_um


def load_glass_ior_cauchy(nd, abbe):
    lambdas = [656.3, 587.56, 486.1]

    if abbe == 0.0:
        B = 0.0
    else:
        B = (nd - 1) / abbe / ((1 / lambdas[2]**2) - (1 / lambdas[0]**2))
    A = nd - B / lambdas[1]**2

    def ior(w):
        return A + B / w**2
    return ior
