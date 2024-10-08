
def LGN1(M, N, X):
    nonlocal C_LGN1, K_LGN1, B_LGN1, H_LGN1
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



def height(MR,VFlag, GamT, FI , N0, K_vbv, B_vbv):
        C_LGN1 = 0.0
        K_LGN1 = 0
        B_LGN1 = 0.0
        H_LGN1 = 0.0
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

        K_vbv[2][0] = BaseC20 - NormGrav[GamT].C20 / math.sqrt(5)
        K_vbv[4][0] = BaseC40 - NormGrav[GamT].C40 / math.sqrt(9)

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

