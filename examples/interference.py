import itertools
from matplotlib import pyplot as plt
import numpy as np

import pygfunction as gt


def main():
    # -------------------------------------------------------------------------
    # Simulation parameters
    # -------------------------------------------------------------------------

    # Borehole dimensions
    D = 1.0             # Borehole buried depth (m)
    H = 100.0           # Borehole length (m)
    r_b = 0.05          # Borehole radius (m)
    B = 6.0             # Borehole spacing (m)

    # Thermal properties
    alpha = 1.0e-6      # Ground thermal diffusivity (m2/s)

    # g-Function calculation options
    options = {'nSegments': 1,
               'segment_ratios': None,
               'disp': False,
               'kClusters': 0}

    # Geometrically expanding time vector.
    dt = 100*3600.                  # Time step
    tmax = 3000. * 8760. * 3600.    # Maximum time
    Nt = 25                         # Number of time steps
    ts = H**2/(9.*alpha)            # Bore field characteristic time
    time = gt.utilities.time_geometric(dt, tmax, Nt)
    lntts = np.log(time/ts)

    # -------------------------------------------------------------------------
    # Borehole fields
    # -------------------------------------------------------------------------

    # Field sizes (fields are side to side with the same initial spacing)
    N_1 = np.array(
        [6, 8, 6],
        dtype=np.uint)
    N_2 = np.array(
        [18, 18, 18],
        dtype=np.uint)
    # N_1 = np.full(
    #     20, 1,
    #     dtype=np.uint)
    # N_2 = np.full(
    #     20, 18,
    #     dtype=np.uint)

    # Borefields
    fields = [
        gt.boreholes.rectangle_field(n_1, n_2, B, B, H, D, r_b)
        for (n_1, n_2) in zip(N_1, N_2)
        ]
    # Move fields and adjust radius to split fields
    for i, field in enumerate(fields):
        for b in field:
            b.x = b.x + B * np.sum(N_1[:i])
            b.r_b = b.r_b * 1.0001**i
    nBoreholes = np.array(
        [len(field) for field in fields],
        dtype=np.uint)
    nFields = len(fields)

    # Complete list of boreholes
    boreholes = list(itertools.chain.from_iterable(fields))
    gt.boreholes.visualize_field(boreholes, view3D=False)

    # -------------------------------------------------------------------------
    # Evaluate g-function for entire field (for verification)
    # -------------------------------------------------------------------------
    gfunc = gt.gfunction.gFunction(
        boreholes, alpha, time=time, options=options,
        method='equivalent', boundary_condition='UHTR')

    # -------------------------------------------------------------------------
    # Evaluate self- and cross- g-functions for sub-fields
    # -------------------------------------------------------------------------

    # Number of equivalent boreholes in sub-fields
    n1 = np.cumsum(nBoreholes)
    cumsum = np.insert(np.cumsum(nBoreholes), 0, 0)
    n0 = np.concatenate(([0], n1[:-1]))
    nEqBoreholes = np.array(
        [len(np.unique(gfunc.solver.clusters[n0[i]:n1[i]]))
         for i, field in enumerate(fields)],
        dtype=np.uint)
    n1 = np.cumsum(nEqBoreholes)
    n0 = np.concatenate(([0], n1[:-1]))
    nBoreholes_eq = [
        np.array(
            [b.nBoreholes
             for b in gfunc.solver.boreholes[n0[i]:n1[i]]]
            )
        for i, field in enumerate(fields)
        ]

    h_ij = gfunc.solver.thermal_response_factors(time, alpha).y[:,:,1:]
    g_ij = np.zeros((nFields, nFields, Nt))
    g_tot = np.zeros(Nt)
    for k in range(Nt):
        for i, nBoreholes_i in enumerate(nBoreholes_eq):
            for j, nBoreholes_j in enumerate(nBoreholes_eq):
                g_ij[i, j, k] = nBoreholes_i @ np.sum(h_ij[n0[i]:n1[i], n0[j]:n1[j], k], axis=1) / np.sum(nBoreholes_i)
        g_tot[k] = (nBoreholes @ np.sum(g_ij[:, :, k], axis=1)) / np.sum(nBoreholes)

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------
    ax = gfunc.visualize_g_function().axes[0]
    for i, nBoreholes_i in enumerate(nBoreholes_eq):
        for j, nBoreholes_j in enumerate(nBoreholes_eq):
            ax.plot(lntts, g_ij[i, j, :], label=f'g({j}->{i})')
    ax.plot(lntts, g_tot, 'k.', label='Total')
    ax.legend()
    plt.title('UHTR + UHTR for cross g-functions')
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Evaluate g-function for entire field and sub-fields using UBWT
    # -------------------------------------------------------------------------
    options_BWT = {
        'nSegments': 8,
        'disp': False,
        'kClusters': 1,
        'profiles': True}
    gfunc_UBWT = gt.gfunction.gFunction(
        boreholes, alpha, time=time, options=options_BWT,
        method='equivalent', boundary_condition='UBWT')

    g_ij_UBWT = g_ij[:, :, :] # Indices are included to make a copy of the object
    for i, field in enumerate(fields):
        g_ij_UBWT[i, i, :] = gt.gfunction.gFunction(
            field, alpha, time=time, options=options_BWT,
            method='equivalent', boundary_condition='UBWT').gFunc
    g_tot_UBWT = np.zeros(Nt)
    for k in range(Nt):
        g_tot_UBWT[k] = (nBoreholes @ np.sum(g_ij_UBWT[:, :, k], axis=1)) / np.sum(nBoreholes)

    # -------------------------------------------------------------------------
    # Plot g-functions
    # -------------------------------------------------------------------------
    ax = gfunc_UBWT.visualize_g_function().axes[0]
    for i, nBoreholes_i in enumerate(nBoreholes_eq):
        for j, nBoreholes_j in enumerate(nBoreholes_eq):
            ax.plot(lntts, g_ij_UBWT[i, j, :], label=f'g({j}->{i})')
    ax.plot(lntts, g_tot_UBWT, 'k.', label='Total')
    ax.legend()
    plt.title('UBWT + UHRT for cross g-functions')
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Evaluate g-function for entire field based on UBWT
    # -------------------------------------------------------------------------
    def get_Qb(field, time, cluster, offset, i):
        gfunc = gt.gfunction.gFunction(field, alpha, time=time, options=options_BWT, method='equivalent', boundary_condition='UBWT')
        # get Q_b in equivalent borehole system of particular field
        z, Q_b = gfunc._heat_extraction_rate_profiles(time=None, iBoreholes=range(len(gfunc.solver.boreholes)))
        # convert Q_b to equivalent borehole system of total field
        Q_b_new = np.zeros((len(cluster), len(Q_b[0])))
        # iterate over all the boreholes in the entire field
        for idx, borehole in enumerate(gfunc_UBWT.solver.clusters[cumsum[i]: cumsum[i+1]]):
            old_eq_borehole = gfunc.solver.clusters[idx][0]
            new_eq_borehole = borehole[0] - offset
            # divide by total number of boreholes in cluster[old_eq_borehole], in order to average the Q_b out
            cluster = [b.nBoreholes for b in gfunc.solver.boreholes]
            Q_b_new[new_eq_borehole, :] += Q_b[old_eq_borehole] / cluster[old_eq_borehole]

        return np.array(Q_b_new).flatten()

    # Clusters are defined in another way when using UBWT for entire field
    # Number of equivalent boreholes in sub-fields
    n1 = np.cumsum(nBoreholes)
    n0 = np.concatenate(([0], n1[:-1]))
    nEqBoreholes = np.array(
        [len(np.unique(gfunc_UBWT.solver.clusters[n0[i]:n1[i]]))
         for i, field in enumerate(fields)],
        dtype=np.uint)
    n1 = np.cumsum(nEqBoreholes)
    n0 = np.concatenate(([0], n1[:-1]))
    nBoreholes_eq = [
        np.array(
            [b.nBoreholes
             for b in gfunc_UBWT.solver.boreholes[n0[i]:n1[i]]]
        )
        for i, field in enumerate(fields)
    ]

    h_ij = gfunc_UBWT.solver.thermal_response_factors(time, alpha).y[:, :, 1:]

    g_ij_UBWT_full = g_ij_UBWT[:, :, :]  # Indices are included to make a copy of the object
    for k in range(Nt):
        for i, nSegments_i in enumerate(nBoreholes_eq):
            for j, nSegments_j in enumerate(nBoreholes_eq):
                if i == j:
                    # only change cross g-functions
                    continue
                # g (j -> i)
                Q_b = get_Qb(fields[j], time[k], nSegments_j, n0[j], j)
                # calculate temperature difference for all segments
                nSegments = options_BWT['nSegments']
                delta_T = h_ij[n0[i]*nSegments:n1[i]*nSegments, n0[j]*nSegments:n1[j]*nSegments, k] @ Q_b
                coeff = np.array([[nb_of_segments/nSegments]*nSegments for nb_of_segments in nSegments_i]).flatten()
                # calculate average borehole wall temperature of equivalent borehole
                g_ij_UBWT_full[i, j, k] = coeff @ delta_T / np.sum(nSegments_i)

    g_tot_UBWT_full = np.zeros(Nt)
    g_tot_UBWT_full2 = np.zeros(Nt)
    for k in range(Nt):
        g_tot_UBWT_full[k] = (nBoreholes @ np.sum(g_ij_UBWT_full[:, :, k], axis=1)) / np.sum(nBoreholes)
        A = np.block(
            [[g_ij_UBWT_full[:, :, k], -np.ones((nFields, 1))],
              [np.atleast_2d(nBoreholes), np.zeros((1, 1))]])
        B = np.zeros(nFields + 1)
        B[-1] = np.sum(nBoreholes)
        X = np.linalg.solve(A, B)
        g_tot_UBWT_full2[k] = X[-1]

    ax = gfunc_UBWT.visualize_g_function().axes[0]
    for i, nBoreholes_i in enumerate(nBoreholes_eq):
        for j, nBoreholes_j in enumerate(nBoreholes_eq):
            ax.plot(lntts, g_ij_UBWT_full[i, j, :], label=f'g({j}->{i})')
    ax.plot(lntts, g_tot_UBWT_full, 'k.', label='Total')
    ax.legend()
    plt.title('UBWT + UBWT for cross g-functions')
    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Comparison of different methods for cross g-functions
    # -------------------------------------------------------------------------
    ax = gfunc.visualize_g_function().axes[0]
    ax.plot(lntts, g_tot, label='UHTR total field')
    ax.plot(lntts, g_tot, marker='.', label='UHTR + UHTR for cross g-functions')
    ax.plot(lntts, g_tot_UBWT, marker='.', label='UBWT + UHRT for cross g-functions')
    ax.plot(lntts, g_tot_UBWT_full, marker='.', label='UBWT + UBWT for cross g-functions')
    ax.plot(lntts, g_tot_UBWT_full2, marker='.', label='UBWT + UBWT for cross g-functions (2)')
    ax.plot(lntts, gfunc_UBWT.gFunc, label='UBWT total field')
    ax.legend()
    plt.title('Comparison of different methods for cross g-functions')
    plt.tight_layout()

    plt.show()

    return h_ij, g_ij


if __name__ == '__main__':
    x = main()
