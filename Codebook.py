from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import math
import torch


@dataclass
class Codebook:
    """
    Generic beam codebook container.

    matrix shape:
        [num_ant, num_beams]

    각 column이 하나의 beamforming vector.
    DFT, learned, hierarchical, random codebook 모두 이 형태로 감싸서 사용한다.
    """

    matrix: torch.Tensor
    name: str = "custom"
    array_type: str = "unknown"
    grid: dict[str, torch.Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.matrix.ndim != 2:
            raise ValueError(
                "Codebook matrix must have shape [num_ant, num_beams]. "
                f"Got shape {self.matrix.shape}."
            )

        if not torch.is_complex(self.matrix):
            raise ValueError("Codebook matrix must be a complex tensor.")

    @property
    def num_ant(self) -> int:
        return self.matrix.shape[0]

    @property
    def num_beams(self) -> int:
        return self.matrix.shape[1]

    @property
    def device(self):
        return self.matrix.device

    @property
    def dtype(self):
        return self.matrix.dtype

    def beam(self, index: int) -> torch.Tensor:
        """
        특정 beam vector 반환.

        Returns:
            Shape [num_ant]
        """
        if index < 0 or index >= self.num_beams:
            raise IndexError(f"Beam index must be in [0, {self.num_beams - 1}].")

        return self.matrix[:, index]

    def select(self, indices: list[int] | torch.Tensor) -> "Codebook":
        """
        일부 beam만 선택한 새로운 Codebook 반환.
        P-2/P-3 refinement에서 후보 beam subset을 만들 때 사용 가능.
        """
        indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)

        return Codebook(
            matrix=self.matrix[:, indices],
            name=f"{self.name}_subset",
            array_type=self.array_type,
            grid=self.grid,
            meta={
                **self.meta,
                "selected_indices": indices.detach().cpu(),
                "parent_name": self.name,
            },
        )

    def to(
        self,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "Codebook":
        """
        device 또는 dtype 변경.
        """
        matrix = self.matrix

        if device is not None or dtype is not None:
            matrix = matrix.to(
                device=device if device is not None else matrix.device,
                dtype=dtype if dtype is not None else matrix.dtype,
            )

        grid = {
            key: value.to(device=device if device is not None else value.device)
            for key, value in self.grid.items()
        }

        return Codebook(
            matrix=matrix,
            name=self.name,
            array_type=self.array_type,
            grid=grid,
            meta=self.meta,
        )

    def check(self) -> dict[str, Any]:
        """
        Codebook sanity check.
        """
        norms = torch.linalg.norm(self.matrix, dim=0)

        result = {
            "name": self.name,
            "array_type": self.array_type,
            "shape": tuple(self.matrix.shape),
            "num_ant": self.num_ant,
            "num_beams": self.num_beams,
            "max_norm_error": torch.max(torch.abs(norms - 1.0)).item(),
        }

        if self.num_ant == self.num_beams:
            gram = self.matrix.conj().T @ self.matrix
            eye = torch.eye(
                self.num_beams,
                device=self.device,
                dtype=self.dtype,
            )
            result["max_orthogonality_error"] = torch.max(
                torch.abs(gram - eye)
            ).item()
        else:
            result["max_orthogonality_error"] = None

        return result

    @classmethod
    def from_tensor(
        cls,
        matrix: torch.Tensor,
        name: str = "custom",
        array_type: str = "unknown",
        grid: dict[str, torch.Tensor] | None = None,
        meta: dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> "Codebook":
        """
        외부에서 만든 임의의 codebook tensor를 Codebook 객체로 감싸기.

        예:
            learned_codebook = Codebook.from_tensor(W, name="learned")
        """
        if normalize:
            matrix = matrix / torch.linalg.norm(matrix, dim=0, keepdim=True).clamp_min(1e-12)

        return cls(
            matrix=matrix,
            name=name,
            array_type=array_type,
            grid={} if grid is None else grid,
            meta={} if meta is None else meta,
        )

    @classmethod
    def dft_ula(
        cls,
        num_ant: int,
        num_beams: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> "Codebook":
        """
        ULA DFT codebook 생성.

        matrix shape:
            [num_ant, num_beams]
        """
        if num_beams is None:
            num_beams = num_ant

        real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64

        n = torch.arange(num_ant, device=device, dtype=real_dtype)[:, None]
        m = torch.arange(num_beams, device=device, dtype=real_dtype)[None, :]

        # Spatial frequency grid: [-1, 1)
        spatial_freq = -1.0 + 2.0 * m / num_beams

        # f_m[n] = exp(j*pi*n*u_m) / sqrt(N_ant)
        phase = math.pi * n * spatial_freq
        matrix = torch.exp(1j * phase) / math.sqrt(num_ant)
        matrix = matrix.to(dtype)

        return cls(
            matrix=matrix,
            name="dft_ula",
            array_type="ULA",
            grid={
                "spatial_freq": spatial_freq.squeeze(0),
            },
            meta={
                "num_ant": num_ant,
                "num_beams": num_beams,
                "spacing": "lambda/2",
            },
        )

    @classmethod
    def dft_upa(
        cls,
        num_ant_x: int,
        num_ant_y: int,
        num_beams_x: int | None = None,
        num_beams_y: int | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.complex64,
    ) -> "Codebook":
        """
        UPA 2D DFT codebook 생성.

        matrix shape:
            [num_ant_x*num_ant_y, num_beams_x*num_beams_y]
        """
        if num_beams_x is None:
            num_beams_x = num_ant_x

        if num_beams_y is None:
            num_beams_y = num_ant_y

        cb_x = cls.dft_ula(
            num_ant=num_ant_x,
            num_beams=num_beams_x,
            device=device,
            dtype=dtype,
        )

        cb_y = cls.dft_ula(
            num_ant=num_ant_y,
            num_beams=num_beams_y,
            device=device,
            dtype=dtype,
        )

        Fx = cb_x.matrix
        Fy = cb_y.matrix

        beams = []

        for iy in range(Fy.shape[1]):
            for ix in range(Fx.shape[1]):
                # UPA codebook = kron(y-axis beam, x-axis beam)
                beam = torch.kron(Fy[:, iy], Fx[:, ix])
                beams.append(beam)

        matrix = torch.stack(beams, dim=1)

        return cls(
            matrix=matrix,
            name="dft_upa",
            array_type="UPA",
            grid={
                "ux": cb_x.grid["spatial_freq"],
                "uy": cb_y.grid["spatial_freq"],
            },
            meta={
                "num_ant_x": num_ant_x,
                "num_ant_y": num_ant_y,
                "num_ant": num_ant_x * num_ant_y,
                "num_beams_x": num_beams_x,
                "num_beams_y": num_beams_y,
                "num_beams": num_beams_x * num_beams_y,
                "spacing_x": "lambda/2",
                "spacing_y": "lambda/2",
            },
        )