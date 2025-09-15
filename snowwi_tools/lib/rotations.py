# Generation of rotation matrices

import numpy as np


def Rx(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])


def Ry(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def Rz(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


def R_ecef_to_ned(lat_deg, lon_deg):
    """Rows are N,E,D axes expressed in ECEF."""
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    sL, cL = np.sin(lat), np.cos(lat)
    sO, cO = np.sin(lon), np.cos(lon)
    return np.array([
        [-sL*cO, -sL*sO,  cL],
        [   -sO,     cO, 0.0],
        [-cL*cO, -cL*sO, -sL]
    ])


def R_body_to_ned(heading_deg, pitch_deg, roll_deg):
    """Aerospace z-y-x (yaw-pitch-roll) from body to NED."""
    psi = np.deg2rad(heading_deg)  # yaw about +D
    th = np.deg2rad(pitch_deg)    # pitch about +E
    phi = np.deg2rad(roll_deg)     # roll about +N
    return Rz(psi) @ Ry(th) @ Rx(phi)


def R_ned_to_tcn_from_velocity(v_ned, heading_deg=None, eps=1e-9):
    """
    Build TCN where:
      T = unit ground-track (horizontal projection of v_ned)
      C = right cross-track = D x T  (D=[0,0,1] in NED)
      D = nadir = [0,0,1]
    Returns R_n2t with rows [T; C; D] in NED components.
    """
    v_h = v_ned.copy()
    v_h[2] = 0.0  # horizontal component
    nh = np.linalg.norm(v_h)
    if nh < eps:
        # fallback: use heading if provided, else point T to North
        if heading_deg is not None:
            χ = np.deg2rad(heading_deg)
            T = np.array([np.cos(χ), np.sin(χ), 0.0])
        else:
            T = np.array([1.0, 0.0, 0.0])
    else:
        T = v_h / nh
    D = np.array([0.0, 0.0, 1.0])
    C = np.cross(D, T)               # right of track
    C /= (np.linalg.norm(C) + eps)
    # re-orthogonalize T to ensure exact right-handed ONB
    T = np.cross(C, D)
    T /= (np.linalg.norm(T) + eps)
    R_n2t = np.vstack([T, C, D])     # NED -> TCN
    return R_n2t


def build_rotation_matrices(lat_deg, lon_deg,
                            heading_deg, pitch_deg, roll_deg,
                            v_ecef):
    """
    Inputs:
      lat_deg, lon_deg : geodetic latitude/longitude (deg)
      heading_deg, pitch_deg, roll_deg : aircraft attitude in degrees
      v_ecef : 3-vector velocity in ECEF (m/s) used to get movement direction

    Returns dict with:
      R_b2n : body -> NED
      R_n2t : NED -> TCN
      R_b2t : body -> TCN
      R_t2n : TCN -> NED
      R_e2n : ECEF -> NED
      R_n2e : NED  -> ECEF
      R_b2e : body -> ECEF
      track_deg : course-over-ground (deg) used to define TCN
    """
    # ECEF -> NED
    R_e2n = R_ecef_to_ned(lat_deg, lon_deg)
    R_n2e = R_e2n.T

    # Velocity to NED to derive track
    v_ned = R_e2n @ v_ecef
    # (Optional) compute track angle χ (course over ground)
    cog = np.rad2deg(np.arctan2(v_ned[1], v_ned[0]))  # atan2(v_E, v_N), in degrees

    # Body -> NED from attitude
    R_b2n = R_body_to_ned(heading_deg, pitch_deg, roll_deg)

    # NED -> TCN from movement direction
    R_n2t = R_ned_to_tcn_from_velocity(v_ned, heading_deg=heading_deg)

    # Compose as needed
    R_b2t = R_n2t @ R_b2n
    R_t2n = R_n2t.T
    R_b2e = R_n2e @ R_b2n

    return dict(
        R_b2n=R_b2n,
        R_n2t=R_n2t,
        R_b2t=R_b2t,
        R_t2n=R_t2n,
        R_e2n=R_e2n,
        R_n2e=R_n2e,
        R_b2e=R_b2e,
        track_deg=cog
    )


if __name__ == "__main__":
    # --- Dummy platform state ---
    lat_deg, lon_deg = 37.0, -122.0          # somewhere over CA
    heading_deg, pitch_deg, roll_deg = 90.0, 2.0, 0.0  # yaw east, slight nose-up, wings level

    # Choose an eastward ground velocity of 80 m/s in NED, then convert to ECEF for the API
    R_e2n = R_ecef_to_ned(lat_deg, lon_deg)
    R_n2e = R_e2n.T
    v_ned_true = np.array([0.0, 80.0, 0.0])  # [N, E, D] m/s: purely east, no climb
    v_ecef = R_n2e @ v_ned_true

    # Build all rotation matrices
    mats = build_rotation_matrices(
        lat_deg, lon_deg,
        heading_deg, pitch_deg, roll_deg,
        v_ecef
    )

    # --- Orthonormality checks ---
    def assert_orthonormal(R, name):
        I = R @ R.T
        assert np.allclose(I, np.eye(3), atol=1e-10), f"{name} is not orthonormal"

    for name in ["R_b2n", "R_n2t", "R_e2n"]:
        assert_orthonormal(mats[name], name)

    # Composition consistency: b->e equals (b->t)->(t->n)->(n->e)
    R_b2e_chain = mats["R_n2e"] @ mats["R_t2n"] @ mats["R_b2t"]
    assert np.allclose(R_b2e_chain, mats["R_b2e"], atol=1e-10), "Chain composition mismatch"

    # --- Transform a sample look vector: left-broadside, 45° depression (down from horizon) in BODY ---
    delta = np.deg2rad(45.0)
    ell_b = np.array([0.0, -np.cos(delta), np.sin(delta)])  # [i, j, k] = [0, -cos, sin]

    # Direct body->ECEFv
    ell_e_direct = mats["R_b2e"] @ ell_b

    # Expanded chain: BODY -> TCN -> NED -> ECEF
    ell_t = mats["R_b2t"] @ ell_b
    ell_n = mats["R_t2n"] @ ell_t
    ell_e_chain = mats["R_n2e"] @ ell_n

    assert np.allclose(ell_e_direct, ell_e_chain, atol=1e-10), "Look vector transform mismatch"

    # --- Sanity on TCN: with eastward motion, Track should align with +E (right), Crosstrack ~ 0, Nadir ~ 0
    v_tcn = mats["R_n2t"] @ v_ned_true
    horiz_speed = np.linalg.norm(v_ned_true[:2])

    assert np.isclose(v_tcn[0], horiz_speed, atol=1e-8), "Track axis magnitude wrong"
    assert np.isclose(v_tcn[1], 0.0, atol=1e-10), "Crosstrack should be ~0 for pure east motion"
    assert np.isclose(v_tcn[2], 0.0, atol=1e-10), "Nadir component should be ~0 for level flight"

    # --- Print a few values ---
    print("All rotation matrix tests passed.")
    print(f"Computed course-over-ground (deg): {mats['track_deg']:.3f}")   # should be ~90°
    print("ECEF look vector (ell_e):", ell_e_direct)
