import math



C_LGN1 = 1
K_LGN1 = 1
B_LGN1 = 1
H_LGN1 = 1
def LGN1(M: int, N: int, X):
    global C_LGN1, K_LGN1, B_LGN1, H_LGN1
    if N == 0:
        C_LGN1 = math.sqrt(0.5)
        K_LGN1 = 0
        B_LGN1 = 0.0
        H_LGN1 = C_LGN1
        return C_LGN1
    else:
        if M > K_LGN1:
            C_LGN1 = (2 * N + 1) * (1 - X * X) / (2 * M)
            C_LGN1 = math.sqrt(C_LGN1) * H_LGN1
            K_LGN1 = M
            B_LGN1 = 0
            H_LGN1 = C_LGN1
            return C_LGN1
        else:
            A = 2 * N
            S = math.sqrt(A * A - 1)
            U = N * N - M * M
            F = 1 / math.sqrt(U)
            P = math.sqrt(abs((A + 1) / (A - 3) * ((N - 1) * (N - 1) - M * M)))
            D = (X * C_LGN1 * S - B_LGN1 * P) * F
            B_LGN1 = C_LGN1
            C_LGN1 = D
            return C_LGN1

NormGrav = [
    {"GamE": 978030.00000, "Bt": 0.0053020, "Bt1": 0.0000070, "C20": -1083.46E-06, "C40": 2.72E-06, "aE": 0},
    {"GamE": 978049.00000, "Bt": 0.0052884, "Bt1": 0.0000059, "C20": -1091.87E-06, "C40": 2.42E-06, "aE": 6378172},
    {"GamE": 978031.80000, "Bt": 0.0053024, "Bt1": 0.0000059, "C20": -1082.78E-06, "C40": 2.37E-06, "aE": 6378172},
    {"GamE": 978031.85000, "Bt": 0.0053024, "Bt1": 0.0000059, "C20": -1083.63E-06, "C40": 1.62E-06, "aE": 6378172},
    {"GamE": 978032.53359, "Bt": 0, "Bt1": 0, "C20": -1082.63E-06, "C40": 2.37E-06, "aE": 6371008.7714, "e": 8.1819190842622E-2}
]

from read_harmonics import read_gfc
K_vbv = read_gfc()

BaseC20  = K_vbv[2, 0]
BaseC40  = K_vbv[4, 0]


def NormGamm(GT: int, S2Fi: float, S22Fi: float) -> float:
    return NormGrav[GT]["GamE"] * (1 + NormGrav[GT]['Bt'] * S2Fi
                                   - NormGrav[GT]['Bt1'] * S22Fi)



# VBV4(MidRad,VKGFlag,GammTyp,fi,DegToRad(OutData[mainI].lb),N0,Koef,OutData[mainI].h);
def height(MR,VFlag, GamT, FI ,AL, N0):
        # C_LGN1 = math.sqrt(0.5)
        # K_LGN1 = 0
        # B_LGN1 = 0.0
        # H_LGN1 = C_LGN1
        RadiusGamm = 1.0
        if VFlag:
            RadiusGamm = MR
        sinFI = math.sin(FI)
        SQRT2 = math.sqrt(2)
        SQRT05 = math.sqrt(0.5)
        S2Fi = sinFI * sinFI
        S22Fi = math.sin(2 * FI) * math.sin(2 * FI)
        GMFi = NormGamm(GamT, S2Fi, S22Fi)

        if GamT == 4:
            GMFi = NormGrav[GamT].GamE * (1 + 0.001931852654 * math.sin(FI) * math.sin(FI)) / math.sqrt(
                1 - 0.0066943799 * math.sin(FI) * math.sin(FI))
            FI = math.atan(math.tan(FI) * (1 - (1 / 298.257)) * (1 - (1 / 298.257)))
            sinFI = math.sin(FI)

        koef[2][0] = BaseC20 - NormGrav[GamT]["C20"] / math.sqrt(5)
        koef[4][0] = BaseC40 - NormGrav[GamT]["C40"] / math.sqrt(9)

        B_vbv = 0.0

        for m in range(N0 + 1):
            R0 = 0
            R1 = 0
            CA = math.cos(m * AL)
            SA = math.sin(m * AL)
            for n in range(m, N0 + 1):
                P1 = LGN1(m, n, sinFI)
                if m == 0:
                    P1 *= SQRT2
                else:
                    P1 *= 2
                if n < 2:
                    continue
                if VFlag:
                    P1 *= RadiusGamm
                else:
                    P1 *= GMFi * (n - 1)
                if m != 0:
                    R1 += K_vbv[m - 1][n] * P1
                R0 += K_vbv[n][m] * P1

            B_vbv += CA * R0 + SA * R1

        K_vbv[2][0] = BaseC20
        K_vbv[4][0] = BaseC40
        return B_vbv

if __name__ == '__main__':
    from read_harmonics import read_gfc

    koef =  read_gfc()
    FI   = -math.pi*0.5
    AL   =  0.0
    MR   =  6378136.46 # m
    GamT = 0

    #VBV4     (MidRad,VKGFlag,GammTyp,fi,DegToRad(OutData[mainI].lb),N0,           Koef,OutData[mainI].h);
    h = height(MR,    True,   GamT,   FI ,AL,                       koef.shape[0])