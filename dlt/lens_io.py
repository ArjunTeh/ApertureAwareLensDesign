import numpy as np

from . import optical_properties
from . import system_matrix
from . import zemax_loader


def clamp_semidiameter(lens):
    k1 = lens[:-1, 0]
    k2 = lens[1:, 0]
    th = lens[:-1, 1]
    max_height = optical_properties.effective_semidiameter(k1, k2, th)

    lens_clamp = np.array(lens)
    lens_clamp[:-1, 3] = np.minimum(lens[:-1, 3], max_height)
    lens_clamp[1:, 3] = np.minimum(lens[1:, 3], max_height)
    return lens_clamp



def export_lens(fname, lens, lens_desc_generic=None, wavelengths=[587.56], distances=None, surf_ids=None):

    if lens_desc_generic is None:
        lens_type_list = ['STANDARD']*lens.shape[0]
    else:
        lens_type_list = [s['surf_type'] for s in lens_desc_generic[1:-1]]

    asphere = zemax_loader.ZemaxSurfaceType.EVENASPH in lens_type_list

    if wavelengths is None:
        wavelengths = [486.1, 587.56, 656.3]

    iors = [
        [s['ior_fn'](w) 
         for s in lens_desc_generic[1:-1]]
        for w in wavelengths
    ]
    iors = np.array(iors)
    
    right_vertex = lens[:, 1].sum()
    left_vertex = 0

    if lens_desc_generic is None:
        mat, mat_list = system_matrix.zemax_state_to_system_matrix(lens, keep_list=True)
        distances = None
        surf_ids = None
    else:
        if distances is None or surf_ids is None:
            _, _, (surf_ids, distances) = zemax_loader.desc_state_to_array(lens_desc_generic, config_id=-1)

        lens_first = np.array(lens)
        if np.any(distances):
            lens_first[surf_ids, 1] = distances[0]
        effl, ebfl = optical_properties.effective_focal_length_from_state(lens_first, asphere=asphere)

    focal_length = ebfl
    back_plane_offset = -ebfl
    # focal_length = effl
    front_plane_offset = effl

    # make sure that the semidiameter matches the constraints of the lens
    lens = clamp_semidiameter(lens)

    dict_out = dict(
        lens=lens,
        iors=iors,
        wavelengths=wavelengths,
        effective_focal_length=focal_length,
        back_plane=right_vertex + back_plane_offset,
        front_plane=left_vertex + front_plane_offset,
        lens_desc=np.array(lens_type_list),
        surf_ids=surf_ids,
        distances=distances,
    )

    np.savez(fname, **dict_out)