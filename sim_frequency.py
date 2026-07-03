from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

import sionna.phy as snphy  # noqa: F401

from Codebook import Codebook
from BeamSweeping import p1_initial_sweep


# =========================
# Simulation configuration
# =========================

SEED = 2026
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.complex64
REAL_DTYPE = torch.float32

C = 299_792_458.0

BANDS = {
    "FR1 3.5 GHz": 3.5e9,
    "FR3 14 GHz": 14e9,
    "FR2 28 GHz": 28e9,
}

NUM_TX_ANT = 16
NUM_RX_ANT = 8

NUM_TX_BEAMS = 16
NUM_RX_BEAMS = 8

NUM_PATHS = 3
DISTANCE_M = 30.0

# 대역 비교에서는 path loss 차이가 보여야 하므로 normalize=False.
NORMALIZE_CHANNEL = False

# path loss 포함 시 일반적인 -10~20 dB에서는 BER이 전부 나빠질 수 있음.
# 그래서 input SNR before path loss 기준으로 높게 설정.
SNR_DB_LIST = torch.arange(60, 141, 5, dtype=REAL_DTYPE)
EFFECTIVE_SNR_REF_DB = 100.0

NUM_TRIALS = 500
NUM_QPSK_SYMBOLS = 1024

OUT_DIR = Path("results_band_compare_sweeping")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Helpers
# =========================

def get_codebook_matrix(codebook):
    if hasattr(codebook, "matrix"):
        return codebook.matrix

    if isinstance(codebook, torch.Tensor):
        return codebook

    raise TypeError(
        "codebook must be either a Codebook object with .matrix "
        "or a torch.Tensor."
    )


def make_ula_dft_codebooks(
    num_tx_ant: int,
    num_rx_ant: int,
    num_tx_beams: int,
    num_rx_beams: int,
    device: torch.device,
    dtype: torch.dtype,
):
    tx_codebook = Codebook.dft_ula(
        num_ant=num_tx_ant,
        num_beams=num_tx_beams,
        device=device,
        dtype=dtype,
    )

    rx_codebook = Codebook.dft_ula(
        num_ant=num_rx_ant,
        num_beams=num_rx_beams,
        device=device,
        dtype=dtype,
    )

    F_tx = get_codebook_matrix(tx_codebook).to(device=device, dtype=dtype)
    W_rx = get_codebook_matrix(rx_codebook).to(device=device, dtype=dtype)

    return tx_codebook, rx_codebook, F_tx, W_rx


def complex_normal(shape, device=DEVICE, dtype=DTYPE):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    x = torch.randn(shape, device=device, dtype=real_dtype)
    y = torch.randn(shape, device=device, dtype=real_dtype)
    return (x + 1j * y).to(dtype) / math.sqrt(2.0)


def lin_to_db_np(x: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, eps))


# =========================
# Channel model
# =========================

def steering_vector_ula(
    num_ant: int,
    angle_rad: torch.Tensor,
    fc_hz: float,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
):
    del fc_hz

    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    n = torch.arange(num_ant, device=device, dtype=real_dtype)

    phase = math.pi * n * torch.sin(angle_rad)
    a = torch.exp(1j * phase) / math.sqrt(num_ant)

    return a.to(dtype)


def geometric_mimo_channel(
    num_rx_ant: int,
    num_tx_ant: int,
    num_paths: int,
    fc_hz: float,
    distance_m: float,
    normalize: bool = False,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
):
    """
    Frequency-dependent geometric sparse MIMO channel.
    normalize=False이면 free-space path loss가 반영됨.
    """
    wavelength = C / fc_hz

    H = torch.zeros(
        (num_rx_ant, num_tx_ant),
        device=device,
        dtype=dtype,
    )

    for _ in range(num_paths):
        aoa = (torch.rand((), device=device) - 0.5) * math.pi
        aod = (torch.rand((), device=device) - 0.5) * math.pi

        excess_distance = torch.rand((), device=device) * 0.25 * distance_m
        path_length = distance_m + excess_distance

        alpha = complex_normal((), device=device, dtype=dtype)

        pathloss_amp = wavelength / (4.0 * math.pi * path_length)

        prop_phase_angle = -2.0 * math.pi * path_length / wavelength
        prop_phase = torch.cos(prop_phase_angle) + 1j * torch.sin(prop_phase_angle)
        prop_phase = prop_phase.to(dtype)

        ar = steering_vector_ula(num_rx_ant, aoa, fc_hz, device, dtype)
        at = steering_vector_ula(num_tx_ant, aod, fc_hz, device, dtype)

        H = H + alpha * pathloss_amp * prop_phase * torch.outer(ar, at.conj())

    H = math.sqrt(num_rx_ant * num_tx_ant / num_paths) * H

    if normalize:
        fro = torch.linalg.norm(H, ord="fro").clamp_min(1e-12)
        H = H / fro * math.sqrt(num_rx_ant * num_tx_ant)

    return H


# =========================
# Beam sweeping
# =========================

def effective_channel(H: torch.Tensor, f: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    h_eff = w^H H f
    """
    return torch.sum(w.conj() * (H @ f))


def beam_sweeping_selection(
    H: torch.Tensor,
    F_tx: torch.Tensor,
    W_rx: torch.Tensor,
):
    result = p1_initial_sweep(
        H=H,
        tx_codebook=F_tx,
        rx_codebook=W_rx,
        tx_rs_power=1.0,
        topk=1,
    )

    best_tx_idx = result.best_tx_beam
    best_rx_idx = result.best_rx_beam

    f = F_tx[:, best_tx_idx]
    w = W_rx[:, best_rx_idx]

    return f, w, best_tx_idx, best_rx_idx, result


# =========================
# QPSK link simulation
# =========================

def qpsk_mod(bits: torch.Tensor) -> torch.Tensor:
    b = bits.reshape(-1, 2).to(torch.float32)

    real = 1.0 - 2.0 * b[:, 0]
    imag = 1.0 - 2.0 * b[:, 1]

    x = (real + 1j * imag) / math.sqrt(2.0)
    return x.to(DTYPE)


def qpsk_demod(x_hat: torch.Tensor) -> torch.Tensor:
    bits_hat = torch.empty(
        (x_hat.numel(), 2),
        device=x_hat.device,
        dtype=torch.int64,
    )

    bits_hat[:, 0] = (x_hat.real < 0.0).to(torch.int64)
    bits_hat[:, 1] = (x_hat.imag < 0.0).to(torch.int64)

    return bits_hat.reshape(-1)


def simulate_qpsk_link_counts(
    h_eff: torch.Tensor,
    snr_db: float,
    bits: torch.Tensor,
):
    x = qpsk_mod(bits)

    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_var = 1.0 / snr_lin

    noise = math.sqrt(noise_var / 2.0) * (
        torch.randn_like(x.real) + 1j * torch.randn_like(x.real)
    )
    noise = noise.to(DTYPE)

    y = h_eff * x + noise

    h_safe = h_eff if torch.abs(h_eff) > 1e-12 else h_eff + 1e-12
    x_hat = y / h_safe

    bits_hat = qpsk_demod(x_hat)

    bit_errors = torch.sum(bits_hat != bits).detach().cpu().item()

    bit_error_mask = bits_hat.reshape(-1, 2) != bits.reshape(-1, 2)
    sym_errors = torch.sum(torch.any(bit_error_mask, dim=1)).detach().cpu().item()

    return bit_errors, bits.numel(), sym_errors, x.numel()


# =========================
# Plotting
# =========================

def plot_cdf(data_dict: dict[str, np.ndarray], xlabel: str, title: str, filename: Path):
    plt.figure()

    for label, values in data_dict.items():
        values = np.asarray(values)
        values = np.sort(values)
        y = np.arange(1, len(values) + 1) / len(values)
        plt.plot(values, y, label=label)

    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def plot_ber(
    snr_db_list: np.ndarray,
    ber_dict: dict[str, np.ndarray],
    title: str,
    filename: Path,
):
    plt.figure()

    for label, ber in ber_dict.items():
        plt.semilogy(snr_db_list, ber, marker="o", label=label)

    plt.xlabel("Input SNR before path loss [dB]")
    plt.ylabel("BER")
    plt.title(title)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


# =========================
# Main simulation
# =========================

def main():
    print(f"Device: {DEVICE}")
    print(f"Normalize channel: {NORMALIZE_CHANNEL}")
    print(f"Distance: {DISTANCE_M} m")

    _, _, F_tx, W_rx = make_ula_dft_codebooks(
        num_tx_ant=NUM_TX_ANT,
        num_rx_ant=NUM_RX_ANT,
        num_tx_beams=NUM_TX_BEAMS,
        num_rx_beams=NUM_RX_BEAMS,
        device=DEVICE,
        dtype=DTYPE,
    )

    band_names = list(BANDS.keys())

    beam_gain = {name: [] for name in band_names}

    num_snr = len(SNR_DB_LIST)
    bit_errors = {name: torch.zeros(num_snr, dtype=torch.float64) for name in band_names}
    bit_totals = {name: torch.zeros(num_snr, dtype=torch.float64) for name in band_names}

    ser_errors = {name: torch.zeros(num_snr, dtype=torch.float64) for name in band_names}
    ser_totals = {name: torch.zeros(num_snr, dtype=torch.float64) for name in band_names}

    for trial in range(NUM_TRIALS):
        bits = torch.randint(
            0,
            2,
            (2 * NUM_QPSK_SYMBOLS,),
            device=DEVICE,
            dtype=torch.int64,
        )

        for band_name, fc_hz in BANDS.items():
            H = geometric_mimo_channel(
                num_rx_ant=NUM_RX_ANT,
                num_tx_ant=NUM_TX_ANT,
                num_paths=NUM_PATHS,
                fc_hz=fc_hz,
                distance_m=DISTANCE_M,
                normalize=NORMALIZE_CHANNEL,
                device=DEVICE,
                dtype=DTYPE,
            )

            f_sweep, w_sweep, _, _, _ = beam_sweeping_selection(
                H=H,
                F_tx=F_tx,
                W_rx=W_rx,
            )

            h_eff = effective_channel(H, f_sweep, w_sweep)

            gain = torch.abs(h_eff) ** 2
            beam_gain[band_name].append(float(gain.detach().cpu().item()))

            for si, snr_db_t in enumerate(SNR_DB_LIST):
                snr_db = float(snr_db_t.item())

                be, bt, se, st = simulate_qpsk_link_counts(
                    h_eff=h_eff,
                    snr_db=snr_db,
                    bits=bits,
                )

                bit_errors[band_name][si] += be
                bit_totals[band_name][si] += bt
                ser_errors[band_name][si] += se
                ser_totals[band_name][si] += st

        if (trial + 1) % 50 == 0:
            print(f"Trial {trial + 1}/{NUM_TRIALS}")

    gain_db = {
        name: lin_to_db_np(np.asarray(values))
        for name, values in beam_gain.items()
    }

    eff_snr_db = {
        name: EFFECTIVE_SNR_REF_DB + gain_db[name]
        for name in band_names
    }

    ber = {
        name: (bit_errors[name] / bit_totals[name]).numpy()
        for name in band_names
    }

    ser = {
        name: (ser_errors[name] / ser_totals[name]).numpy()
        for name in band_names
    }

    snr_np = SNR_DB_LIST.numpy()

    plot_cdf(
        gain_db,
        xlabel="Beam sweeping gain |wᴴHf|² [dB]",
        title="Beam sweeping gain CDF across bands",
        filename=OUT_DIR / "band_compare_beam_gain_cdf.png",
    )

    plot_cdf(
        eff_snr_db,
        xlabel=f"Effective SNR at input SNR={EFFECTIVE_SNR_REF_DB:.1f} dB [dB]",
        title="Effective SNR CDF across bands",
        filename=OUT_DIR / "band_compare_effective_snr_cdf.png",
    )

    plot_ber(
        snr_db_list=snr_np,
        ber_dict=ber,
        title="Beam sweeping BER vs SNR across bands",
        filename=OUT_DIR / "band_compare_ber_vs_snr.png",
    )

    print("\nSaved figures:")
    print(OUT_DIR / "band_compare_beam_gain_cdf.png")
    print(OUT_DIR / "band_compare_effective_snr_cdf.png")
    print(OUT_DIR / "band_compare_ber_vs_snr.png")

    print("\nFinal BER:")
    for name in band_names:
        print(name, ber[name])

    print("\nFinal SER:")
    for name in band_names:
        print(name, ser[name])


if __name__ == "__main__":
    main()