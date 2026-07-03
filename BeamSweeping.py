from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class BeamSweepResult:
    """
    Beam sweeping 결과를 저장하는 dataclass.

    Attributes:
        procedure:
            "P1", "P2", "P3" 등 절차 이름.
        power:
            Candidate beam pair별 측정 power.
            Shape [num_rx_candidates, num_tx_candidates]
        tx_beam_indices:
            power의 column이 실제 전체 Tx codebook의 몇 번째 beam인지 나타냄.
        rx_beam_indices:
            power의 row가 실제 전체 Rx codebook의 몇 번째 beam인지 나타냄.
        best_tx_beam:
            선택된 Tx beam index.
        best_rx_beam:
            선택된 Rx beam index.
        best_value:
            선택된 beam pair의 측정값.
        topk_tx_beams:
            상위 k개 Tx beam index.
        topk_rx_beams:
            상위 k개 Rx beam index.
        topk_values:
            상위 k개 측정값.
    """

    procedure: str
    power: torch.Tensor
    tx_beam_indices: torch.Tensor
    rx_beam_indices: torch.Tensor
    best_tx_beam: int
    best_rx_beam: int
    best_value: float
    topk_tx_beams: torch.Tensor
    topk_rx_beams: torch.Tensor
    topk_values: torch.Tensor


def _to_index_tensor(
    indices: Optional[Sequence[int] | torch.Tensor],
    total_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    None이면 전체 index를 사용하고, 아니면 입력 index를 tensor로 변환.
    """

    if indices is None:
        return torch.arange(total_size, device=device, dtype=torch.long)

    index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)

    if torch.any(index_tensor < 0) or torch.any(index_tensor >= total_size):
        raise ValueError(
            f"indices must be in [0, {total_size - 1}], "
            f"but got {index_tensor.detach().cpu().tolist()}"
        )

    return index_tensor

# 추후에 새로운 알고리즘으로 빔 후보를 선택할 수 있음
def neighbor_beam_indices(
    center: int,
    radius: int,
    num_beams: int,
    circular: bool = True,
) -> list[int]:
    """
    P-2/P-3 refinement용 주변 beam index 생성.

    Args:
        center:
            중심 beam index.
        radius:
            center 주변으로 몇 개씩 볼지.
            radius=2이면 center-2, center-1, center, center+1, center+2.
        num_beams:
            전체 beam 개수.
        circular:
            True이면 DFT grid처럼 양끝을 wrap-around.
            False이면 범위를 벗어나는 index는 제거.

    Returns:
        주변 beam index list.
    """

    if radius < 0:
        raise ValueError("radius must be non-negative.")

    if center < 0 or center >= num_beams:
        raise ValueError(f"center must be in [0, {num_beams - 1}].")

    indices = []

    for offset in range(-radius, radius + 1):
        idx = center + offset

        if circular:
            idx = idx % num_beams
            indices.append(idx)
        else:
            if 0 <= idx < num_beams:
                indices.append(idx)

    # 중복 제거 + 정렬
    return sorted(set(indices))


def compute_effective_channel(
    H: torch.Tensor,
    tx_codebook: torch.Tensor,
    rx_codebook: torch.Tensor,
) -> torch.Tensor:
    """
    모든 Tx/Rx beam 조합에 대한 effective channel 계산.

    Args:
        H:
            Channel matrix.
            Shape [..., num_rx_ant, num_tx_ant]
            예:
                [num_rx_ant, num_tx_ant]
                [batch_size, num_rx_ant, num_tx_ant]
                [batch_size, num_subcarriers, num_rx_ant, num_tx_ant]
        tx_codebook:
            Tx beam codebook.
            Shape [num_tx_ant, num_tx_beams]
        rx_codebook:
            Rx beam codebook.
            Shape [num_rx_ant, num_rx_beams]

    Returns:
        H_eff:
            Effective channel.
            Shape [..., num_rx_beams, num_tx_beams]
    """

    if H.ndim < 2:
        raise ValueError("H must have at least 2 dimensions: [..., num_rx_ant, num_tx_ant].")

    num_rx_ant = H.shape[-2]
    num_tx_ant = H.shape[-1]

    if tx_codebook.ndim != 2:
        raise ValueError("tx_codebook must have shape [num_tx_ant, num_tx_beams].")

    if rx_codebook.ndim != 2:
        raise ValueError("rx_codebook must have shape [num_rx_ant, num_rx_beams].")

    if tx_codebook.shape[0] != num_tx_ant:
        raise ValueError(
            f"tx_codebook antenna dimension mismatch. "
            f"H has num_tx_ant={num_tx_ant}, "
            f"but tx_codebook has {tx_codebook.shape[0]}."
        )

    if rx_codebook.shape[0] != num_rx_ant:
        raise ValueError(
            f"rx_codebook antenna dimension mismatch. "
            f"H has num_rx_ant={num_rx_ant}, "
            f"but rx_codebook has {rx_codebook.shape[0]}."
        )

    tx_codebook = tx_codebook.to(device=H.device, dtype=H.dtype)
    rx_codebook = rx_codebook.to(device=H.device, dtype=H.dtype)

    # H_eff[..., j, i] = w_j^H H f_i
    #
    # rx_codebook: [num_rx_ant, num_rx_beams]
    # H:           [..., num_rx_ant, num_tx_ant]
    # tx_codebook: [num_tx_ant, num_tx_beams]
    #
    # output:      [..., num_rx_beams, num_tx_beams]
    H_eff = torch.einsum(
        "nr,...nt,tb->...rb",
        rx_codebook.conj(),
        H,
        tx_codebook,
    )

    return H_eff


def compute_l1_rsrp(
    H: torch.Tensor,
    tx_codebook: torch.Tensor,
    rx_codebook: torch.Tensor,
    tx_rs_power: float = 1.0,
    average_over_leading_dims: bool = True,
) -> torch.Tensor:
    """
    Beam pair별 L1-RSRP에 해당하는 등가 power 계산.

    3GPP에서는 SSB/CSI-RS resource별 측정값을 사용하지만,
    여기서는 Sionna.phy channel H 위에서 등가적으로
    |w^H H f|^2 를 계산한다.

    Args:
        H:
            Shape [..., num_rx_ant, num_tx_ant]
        tx_codebook:
            Shape [num_tx_ant, num_tx_beams]
        rx_codebook:
            Shape [num_rx_ant, num_rx_beams]
        tx_rs_power:
            Reference signal transmit power.
        average_over_leading_dims:
            True이면 batch/subcarrier/time 등 leading dimension을 평균.
            예: H shape [batch, subcarrier, rx, tx]이면
            결과를 [rx_beams, tx_beams]로 평균.

    Returns:
        power:
            Shape [num_rx_beams, num_tx_beams] if average_over_leading_dims=True.
            Otherwise shape [..., num_rx_beams, num_tx_beams].
    """

    H_eff = compute_effective_channel(H, tx_codebook, rx_codebook)

    power = tx_rs_power * torch.abs(H_eff) ** 2

    if average_over_leading_dims and power.ndim > 2:
        leading_dims = tuple(range(power.ndim - 2))
        power = power.mean(dim=leading_dims)

    return power


def _make_result(
    procedure: str,
    power: torch.Tensor,
    tx_beam_indices: torch.Tensor,
    rx_beam_indices: torch.Tensor,
    topk: int,
) -> BeamSweepResult:
    """
    power matrix에서 best beam pair와 top-k beam pair를 선택.
    """

    if power.ndim != 2:
        raise ValueError("power must have shape [num_rx_candidates, num_tx_candidates].")

    num_rx_candidates, num_tx_candidates = power.shape

    if num_rx_candidates == 0 or num_tx_candidates == 0:
        raise ValueError("Candidate beam set must not be empty.")

    k = min(topk, power.numel())

    flat_power = power.reshape(-1)
    topk_values, topk_flat_indices = torch.topk(flat_power, k=k)

    local_rx_indices = topk_flat_indices // num_tx_candidates
    local_tx_indices = topk_flat_indices % num_tx_candidates

    global_rx_indices = rx_beam_indices[local_rx_indices]
    global_tx_indices = tx_beam_indices[local_tx_indices]

    best_rx_beam = int(global_rx_indices[0].detach().cpu().item())
    best_tx_beam = int(global_tx_indices[0].detach().cpu().item())
    best_value = float(topk_values[0].detach().cpu().item())

    return BeamSweepResult(
        procedure=procedure,
        power=power,
        tx_beam_indices=tx_beam_indices,
        rx_beam_indices=rx_beam_indices,
        best_tx_beam=best_tx_beam,
        best_rx_beam=best_rx_beam,
        best_value=best_value,
        topk_tx_beams=global_tx_indices.detach().cpu(),
        topk_rx_beams=global_rx_indices.detach().cpu(),
        topk_values=topk_values.detach().cpu(),
    )


def p1_initial_sweep(
    H: torch.Tensor,
    tx_codebook: torch.Tensor,
    rx_codebook: torch.Tensor,
    tx_candidates: Optional[Sequence[int] | torch.Tensor] = None,
    rx_candidates: Optional[Sequence[int] | torch.Tensor] = None,
    tx_rs_power: float = 1.0,
    topk: int = 1,
) -> BeamSweepResult:
    """
    3GPP NR P-1에 해당하는 초기 beam pair 탐색.

    P-1:
        - gNB/TRP Tx beam sweep
        - UE Rx beam sweep
        - Tx/Rx beam pair 후보 선택

    Args:
        H:
            Shape [..., num_rx_ant, num_tx_ant]
        tx_codebook:
            Shape [num_tx_ant, num_tx_beams]
        rx_codebook:
            Shape [num_rx_ant, num_rx_beams]
        tx_candidates:
            탐색할 Tx beam index. None이면 전체 Tx beam 탐색.
        rx_candidates:
            탐색할 Rx beam index. None이면 전체 Rx beam 탐색.
        tx_rs_power:
            RS transmit power.
        topk:
            상위 몇 개 beam pair를 저장할지.

    Returns:
        BeamSweepResult
    """

    device = H.device

    num_tx_beams = tx_codebook.shape[1]
    num_rx_beams = rx_codebook.shape[1]

    tx_indices = _to_index_tensor(tx_candidates, num_tx_beams, device)
    rx_indices = _to_index_tensor(rx_candidates, num_rx_beams, device)

    tx_cb = tx_codebook[:, tx_indices]
    rx_cb = rx_codebook[:, rx_indices]

    power = compute_l1_rsrp(
        H=H,
        tx_codebook=tx_cb,
        rx_codebook=rx_cb,
        tx_rs_power=tx_rs_power,
        average_over_leading_dims=True,
    )

    return _make_result(
        procedure="P1",
        power=power,
        tx_beam_indices=tx_indices,
        rx_beam_indices=rx_indices,
        topk=topk,
    )


def p2_tx_refinement(
    H: torch.Tensor,
    tx_codebook: torch.Tensor,
    rx_codebook: torch.Tensor,
    fixed_rx_beam: int,
    tx_candidates: Optional[Sequence[int] | torch.Tensor] = None,
    tx_rs_power: float = 1.0,
    topk: int = 1,
) -> BeamSweepResult:
    """
    3GPP NR P-2에 해당하는 gNB/TRP Tx beam refinement.

    P-2:
        - UE Rx beam은 고정
        - gNB/TRP Tx beam 후보를 다시 sweep
        - 더 좋은 Tx beam 선택

    Args:
        H:
            Shape [..., num_rx_ant, num_tx_ant]
        tx_codebook:
            Shape [num_tx_ant, num_tx_beams]
        rx_codebook:
            Shape [num_rx_ant, num_rx_beams]
        fixed_rx_beam:
            고정할 Rx beam index.
        tx_candidates:
            refinement할 Tx beam 후보.
            None이면 전체 Tx beam 탐색.
        tx_rs_power:
            RS transmit power.
        topk:
            상위 몇 개 Tx beam을 저장할지.

    Returns:
        BeamSweepResult
    """

    device = H.device

    num_tx_beams = tx_codebook.shape[1]
    num_rx_beams = rx_codebook.shape[1]

    if fixed_rx_beam < 0 or fixed_rx_beam >= num_rx_beams:
        raise ValueError(f"fixed_rx_beam must be in [0, {num_rx_beams - 1}].")

    tx_indices = _to_index_tensor(tx_candidates, num_tx_beams, device)
    rx_indices = torch.tensor([fixed_rx_beam], device=device, dtype=torch.long)

    tx_cb = tx_codebook[:, tx_indices]
    rx_cb = rx_codebook[:, rx_indices]

    power = compute_l1_rsrp(
        H=H,
        tx_codebook=tx_cb,
        rx_codebook=rx_cb,
        tx_rs_power=tx_rs_power,
        average_over_leading_dims=True,
    )

    return _make_result(
        procedure="P2",
        power=power,
        tx_beam_indices=tx_indices,
        rx_beam_indices=rx_indices,
        topk=topk,
    )


def p3_rx_refinement(
    H: torch.Tensor,
    tx_codebook: torch.Tensor,
    rx_codebook: torch.Tensor,
    fixed_tx_beam: int,
    rx_candidates: Optional[Sequence[int] | torch.Tensor] = None,
    tx_rs_power: float = 1.0,
    topk: int = 1,
) -> BeamSweepResult:
    """
    3GPP NR P-3에 해당하는 UE Rx beam refinement.

    P-3:
        - gNB/TRP Tx beam은 고정
        - UE Rx beam 후보를 다시 sweep
        - 더 좋은 Rx beam 선택

    Args:
        H:
            Shape [..., num_rx_ant, num_tx_ant]
        tx_codebook:
            Shape [num_tx_ant, num_tx_beams]
        rx_codebook:
            Shape [num_rx_ant, num_rx_beams]
        fixed_tx_beam:
            고정할 Tx beam index.
        rx_candidates:
            refinement할 Rx beam 후보.
            None이면 전체 Rx beam 탐색.
        tx_rs_power:
            RS transmit power.
        topk:
            상위 몇 개 Rx beam을 저장할지.

    Returns:
        BeamSweepResult
    """

    device = H.device

    num_tx_beams = tx_codebook.shape[1]
    num_rx_beams = rx_codebook.shape[1]

    if fixed_tx_beam < 0 or fixed_tx_beam >= num_tx_beams:
        raise ValueError(f"fixed_tx_beam must be in [0, {num_tx_beams - 1}].")

    tx_indices = torch.tensor([fixed_tx_beam], device=device, dtype=torch.long)
    rx_indices = _to_index_tensor(rx_candidates, num_rx_beams, device)

    tx_cb = tx_codebook[:, tx_indices]
    rx_cb = rx_codebook[:, rx_indices]

    power = compute_l1_rsrp(
        H=H,
        tx_codebook=tx_cb,
        rx_codebook=rx_cb,
        tx_rs_power=tx_rs_power,
        average_over_leading_dims=True,
    )

    return _make_result(
        procedure="P3",
        power=power,
        tx_beam_indices=tx_indices,
        rx_beam_indices=rx_indices,
        topk=topk,
    )


def nr_p1_p2_p3_beam_management(
    H: torch.Tensor,
    tx_codebook: torch.Tensor,
    rx_codebook: torch.Tensor,
    tx_rs_power: float = 1.0,
    run_p2: bool = True,
    run_p3: bool = True,
    p2_tx_candidates: Optional[Sequence[int] | torch.Tensor] = None,
    p3_rx_candidates: Optional[Sequence[int] | torch.Tensor] = None,
    p2_neighbor_radius: Optional[int] = 2,
    p3_neighbor_radius: Optional[int] = 2,
    circular_neighbors: bool = True,
    topk: int = 1,
) -> dict:
    """
    3GPP NR beam management baseline:
        P-1 -> P-2 -> P-3 순서로 수행.

    이 함수는 codebook 종류에 독립적이다.
    DFT codebook, learned codebook, hierarchical codebook 모두 사용 가능하다.

    Args:
        H:
            Shape [..., num_rx_ant, num_tx_ant]
        tx_codebook:
            Shape [num_tx_ant, num_tx_beams]
        rx_codebook:
            Shape [num_rx_ant, num_rx_beams]
        tx_rs_power:
            RS transmit power.
        run_p2:
            True이면 P-2 Tx refinement 수행.
        run_p3:
            True이면 P-3 Rx refinement 수행.
        p2_tx_candidates:
            P-2에서 탐색할 Tx beam index.
            None이고 p2_neighbor_radius가 주어지면 P-1 best Tx 주변 beam 사용.
        p3_rx_candidates:
            P-3에서 탐색할 Rx beam index.
            None이고 p3_neighbor_radius가 주어지면 현재 best Rx 주변 beam 사용.
        p2_neighbor_radius:
            P-2에서 P-1 best Tx beam 주변 몇 개를 refinement할지.
            None이면 전체 Tx beam 탐색.
        p3_neighbor_radius:
            P-3에서 현재 best Rx beam 주변 몇 개를 refinement할지.
            None이면 전체 Rx beam 탐색.
        circular_neighbors:
            True이면 beam index를 circular하게 wrap-around.
        topk:
            각 절차에서 저장할 상위 beam pair 개수.

    Returns:
        results:
            {
                "p1": BeamSweepResult,
                "p2": BeamSweepResult or None,
                "p3": BeamSweepResult or None,
                "selected_tx_beam": int,
                "selected_rx_beam": int,
                "selected_value": float,
            }
    """

    num_tx_beams = tx_codebook.shape[1]
    num_rx_beams = rx_codebook.shape[1]

    # P-1: 초기 Tx/Rx beam pair 탐색
    p1 = p1_initial_sweep(
        H=H,
        tx_codebook=tx_codebook,
        rx_codebook=rx_codebook,
        tx_candidates=None,
        rx_candidates=None,
        tx_rs_power=tx_rs_power,
        topk=topk,
    )

    selected_tx = p1.best_tx_beam
    selected_rx = p1.best_rx_beam
    selected_value = p1.best_value

    # P-2: gNB/TRP Tx beam refinement
    p2 = None

    if run_p2:
        if p2_tx_candidates is None and p2_neighbor_radius is not None:
            p2_tx_candidates = neighbor_beam_indices(
                center=selected_tx,
                radius=p2_neighbor_radius,
                num_beams=num_tx_beams,
                circular=circular_neighbors,
            )

        p2 = p2_tx_refinement(
            H=H,
            tx_codebook=tx_codebook,
            rx_codebook=rx_codebook,
            fixed_rx_beam=selected_rx,
            tx_candidates=p2_tx_candidates,
            tx_rs_power=tx_rs_power,
            topk=topk,
        )

        selected_tx = p2.best_tx_beam
        selected_value = p2.best_value

    # P-3: UE Rx beam refinement
    p3 = None

    if run_p3:
        if p3_rx_candidates is None and p3_neighbor_radius is not None:
            p3_rx_candidates = neighbor_beam_indices(
                center=selected_rx,
                radius=p3_neighbor_radius,
                num_beams=num_rx_beams,
                circular=circular_neighbors,
            )

        p3 = p3_rx_refinement(
            H=H,
            tx_codebook=tx_codebook,
            rx_codebook=rx_codebook,
            fixed_tx_beam=selected_tx,
            rx_candidates=p3_rx_candidates,
            tx_rs_power=tx_rs_power,
            topk=topk,
        )

        selected_rx = p3.best_rx_beam
        selected_value = p3.best_value

    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "selected_tx_beam": selected_tx,
        "selected_rx_beam": selected_rx,
        "selected_value": selected_value,
    }