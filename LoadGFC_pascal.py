def load_gfc(name: str) -> bool:
    import os

    gauge_main_min_value = 0
    gauge_main_max_value = 150 # model_n0
    gauge_main_progress = 0
    load_gfc_result = False
    eof_ok = False

    try:
        with open(name, 'r') as f:
            for line in f:
                s = line.strip()
                eof_ok = 'end_of_head' in s
                if eof_ok:
                    break

        if not eof_ok:
            return load_gfc_result

        num_i = 0
        for line in f:
            parts = line.split()
            sign = parts[0].strip()
            n = int(parts[1])
            m = int(parts[2])
            cc = float(parts[3])
            ss = float(parts[4])

            if sign not in ['gfc', 'gfct']:
                continue

            koef[n][m] = cc
            if m != 0:
                koef[m - 1][n] = ss

            gauge_main_progress = num_i
            num_i += 1

    except IOError as e:
        return load_gfc_result

    gauge_main_progress = 0
    load_gfc_result = True
    return load_gfc_result

