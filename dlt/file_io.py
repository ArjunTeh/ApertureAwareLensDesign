import re


def verify_zemax_file(fname) -> bool: 
    pass


def get_encoding(fname):
    encoding = 'utf-16'
    with open(fname, "r", encoding='utf-16') as file:
        try:
            line = file.readline()
        except:
            encoding = 'utf-8'
    return encoding


def surface_info_to_system(surface_info):
    info = {}
    for fields in surface_info:
        name = fields[0]
        if name == "TYPE":
            info["type"] = fields[1]

        elif name == "MIRR":
            if int(fields[1]) != 2:
                raise ValueError("Mirrors are explicitly not supported for now")

        elif name in ["CURV", "DISZ", "DIAM", "MEMA", "CONI"]:
            info[name] = float(fields[1])

        elif name == "GLAS":
            info[name] = (fields[1], [float(f) for f in fields[2:]])

        elif name == "PARM":
            info.setdefault(name, []).append(float(fields[2]))

        elif name == "STOP":
            info[name] = True

        elif name in ["THIC", "SDIA"]:
            # config number, value
            info.setdefault(name, []).append((int(fields[2]), float(fields[3])))

    return info


def parse_zemax_file(zmx_file):
    '''Takes in a file object that presumably is already the zmx file

    This will read through the file and load the information into a dict
    this will be more agnostic to the underlying representations that will later be
    used in the optimization.
    '''

    encoding = get_encoding(zmx_file)
    surfaces_raw_info = dict()
    wavelengths = []
    glass_cats = []
    cur_surf = -1
    with open(zmx_file, "r", encoding=encoding) as file:
        while True:
            line = file.readline()
            if not line:
                break

            fields = re.split(r"\s+", line.strip())    

            if fields[0] == "SURF":
                cur_surf = int(fields[1])
                surfaces_raw_info[cur_surf] = []
            
            elif re.match(r"\s", line):
                surfaces_raw_info[cur_surf].append(fields)

            elif fields[0] == "WAVM":
                wavelengths.append(fields[1:])

            elif fields[0] == "GCAT":
                glass_cats = fields[1:]

            elif fields[0] == "SDIA":
                surf_num = int(fields[1])
                surfaces_raw_info[surf_num].append(fields)
            
            elif fields[0] == "THIC":
                surf_num = int(fields[1])
                surfaces_raw_info[surf_num].append(fields)

            else:
                pass

    surfaces = [(k, surface_info_to_system(sr)) for k, sr in surfaces_raw_info.items()]

    return surfaces, wavelengths, glass_cats


def export_zemax_file(lens_info, output_file, ref_zmx_file):
    encoding = get_encoding(ref_zmx_file)

    ref_file = open(ref_zmx_file, 'r', encoding=encoding)
    out_file = open(output_file, 'w', encoding=encoding)

    cur_surf = 0
    while True:
        ref_line = ref_file.readline()
        if not ref_line:
            break

        fields = re.split(r"\s+", ref_line.strip())    

        if fields[0] == "SURF":
            cur_surf = int(fields[1]) - 1

        if (cur_surf < len(lens_info)) and (cur_surf >= 0):
            if fields[0] == "CURV":
                fields[1] = lens_info[cur_surf][0]
                ref_line = "  " + " ".join(fields) + "\n"
            elif fields[0] == "DISZ":
                fields[1] = lens_info[cur_surf][1]
                ref_line = "  " + " ".join(fields) + "\n"
            elif fields[0] == "DIAM":
                fields[1] = lens_info[cur_surf][2]
                ref_line = "  " + " ".join(fields) + "\n"

        if '\n' not in ref_line:
            ref_line += '\n'
        out_file.write(ref_line)

    ref_file.close()
    out_file.close()


def parse_zemax_glass_catalog(ga_file):
    encoding = get_encoding(ga_file)
    glass_raw_info = dict()
    with open(ga_file, "r", encoding=encoding) as file:
        cur_glass_name = ""
        while True:
            line = file.readline()
            if not line:
                break

            fields = re.split(r"\s+", line.strip())    

            if fields[0] == "NM":
                cur_glass_name = fields[1].upper()
                glass_raw_info[cur_glass_name] = dict(nm=fields[2:])
            if fields[0] == 'CD':
                glass_raw_info[cur_glass_name]['cd'] = fields[1:]
            if fields[0] == 'LD':
                glass_raw_info[cur_glass_name]['ld'] = fields[1:]
        
    return glass_raw_info