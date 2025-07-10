import math

import kornia
import lightning as L
import torch
from torch import nn


# Your WarpTransform class (looks good, using your version)
class WarpTransform(nn.Module):
    def __init__(self):
        super().__init__()
        # Parameters
        self.hx = nn.Parameter(torch.randn(()), requires_grad=True)
        self.hy = nn.Parameter(torch.randn(()), requires_grad=True)
        self.hz = nn.Parameter(torch.randn(()), requires_grad=True)
        self.th = nn.Parameter(torch.randn(()), requires_grad=True)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:  
        x = xyt[:, 0]
        y = xyt[:, 1]
        t = xyt[:, 2]

        xy = torch.stack([x, y], dim=0)  # Shape (2, N)
        hxy = torch.stack([self.hx, self.hy], dim=0).unsqueeze(1)  # Shape (2,1)

        cos_theta = torch.cos(self.th)
        sin_theta = torch.sin(self.th)

        # Ensure rot_mat is (2,2)
        rot_mat = torch.stack(
            [torch.stack([cos_theta, -sin_theta]), torch.stack([sin_theta, cos_theta])]
        ).squeeze()  # Squeeze in case parameters had an extra dim initially

        rotated_xy = torch.matmul(rot_mat, xy)
        scaled_rot_xy = (
            self.hz + 1
        ) * rotated_xy
        shifted_scaled_rot_xy = hxy + scaled_rot_xy - xy
        delta_xy = t.unsqueeze(0) * shifted_scaled_rot_xy
        warped = xy - delta_xy
        warped_prime = warped.transpose(0, 1)  # Shape (N, 2)
        output_c_prime = torch.cat([warped_prime, t.unsqueeze(1)], dim=1)

        return output_c_prime


class LitWarpTransform(L.LightningModule):
    def __init__(
        self,
        dvs_native_width: int = 240,  # Defaulting to common DVS sizes
        dvs_native_height: int = 180,
        unit_pixel_size: float = 1.0,
        scaling_factor: float = 0.3,
        alg1_lr: float = 1e-3,  # Added optimizer specific hyperparams here
        alg1_iterations: int = 30,
        alg1_xi: float = 1e-5,
        alg2_iterations: int = 20,
        alg2_perturb_abs: float = 1e-4,
        alg2_xi: float = 1e-6,
    ) -> None:
        """
        Initialize the Lightning Module for optimizing the warp transform parameters.
        ---
        # Parameters
        ---
        dvs_native_width: The native width of the sensor in DVS pixels.
        dvs_native_height: The native height of the sensor in DVS pixels.
        unit_pixel_size: The size of one DVS native pixel in the coordinate units used by
                         the warped event coordinates (x_warped, y_warped).
                         E.g., 1.0 if x_warped is in 'DVS pixel units'.
        scaling_factor: The factor (e.g., 0.3) for '0.3 of DVS pixel size',
        defining bin size relative to unit_pixel_size.
        alg1_*: Hyperparameters for Algorithm 1.
        alg2_*: Hyperparameters for Algorithm 2.
        """
        super().__init__()
        self.save_hyperparameters()  # Good practice for LightningModules

        self.model = WarpTransform()
        self.native_height = dvs_native_height
        self.native_width = dvs_native_width
        self.unit_pixel_size = unit_pixel_size
        self.scaling_factor = scaling_factor

        # output_width/height are the number of bins in the event-count/time image
        self.output_img_width = math.ceil(self.native_width / self.scaling_factor)
        self.output_img_height = math.ceil(self.native_height / self.scaling_factor)

        # dbin is the actual dimension of one bin in the output grid,
        # in the same units as x_warped
        # (if x_warped units are consistent with unit_pixel_size).
        self.dbin = self.scaling_factor * self.unit_pixel_size


    def i_img(self, C_prime: torch.Tensor) -> torch.Tensor:
        x_warped = C_prime[:, 0]
        y_warped = C_prime[:, 1]

        i = torch.floor(x_warped / self.dbin).long()
        j = torch.floor(y_warped / self.dbin).long()

        valid_x_mask = (i >= 0) & (i < self.output_img_width)
        valid_y_mask = (j >= 0) & (j < self.output_img_height)
        valid_event_mask = valid_x_mask & valid_y_mask

        valid_i = i[valid_event_mask]
        valid_j = j[valid_event_mask]

        img = torch.zeros(
            (self.output_img_height, self.output_img_width),
            device=C_prime.device,
            dtype=torch.long,
        )

        if valid_i.numel() > 0:
            lin_idx = valid_j * self.output_img_width + valid_i
            uniq_idx, cts = torch.unique(lin_idx, return_counts=True)

            uj = torch.div(uniq_idx, self.output_img_width, rounding_mode="floor")
            ui = uniq_idx % self.output_img_width
            img[uj, ui] = cts
        return img

    def t_img(self, C_prime: torch.Tensor, event_count_image_I: torch.Tensor) -> torch.Tensor:
        x_warped = C_prime[:, 0]
        y_warped = C_prime[:, 1]
        original_timestamps = C_prime[:, 2]

        i = torch.floor(x_warped / self.dbin).long()
        j = torch.floor(y_warped / self.dbin).long()

        valid_x_mask = (i >= 0) & (i < self.output_img_width)
        valid_y_mask = (j >= 0) & (j < self.output_img_height)
        valid_event_mask = valid_x_mask & valid_y_mask

        valid_i = i[valid_event_mask]
        valid_j = j[valid_event_mask]
        valid_timestamps = original_timestamps[valid_event_mask]

        sum_timestamps_image = torch.zeros(
            (self.output_img_height, self.output_img_width),
            device=C_prime.device,
            dtype=torch.float32,
        )

        if valid_i.numel() > 0:
            linear_indices = valid_j * self.output_img_width + valid_i
            sum_timestamps_image.view(-1).scatter_add_(0, linear_indices, valid_timestamps)

        time_image_T = torch.zeros_like(sum_timestamps_image)
        non_zero_counts_mask = event_count_image_I > 0

        if torch.any(non_zero_counts_mask):  # Ensure there's something to divide
            time_image_T[non_zero_counts_mask] = sum_timestamps_image[
                non_zero_counts_mask
            ] / event_count_image_I[non_zero_counts_mask].type_as(sum_timestamps_image)
        return time_image_T

    def calc_error_metric_eq6(
        self, C_prime_warped: torch.Tensor
    ) -> torch.Tensor:  # Renamed for clarity
        """Calculates the error metric from Eq. 6 (sum of squared gradients of Time-Image)."""
        count_img = self.i_img(C_prime_warped)
        time_img = self.t_img(C_prime_warped, count_img)

        if time_img.ndim == 2:
            time_img_kornia = time_img.unsqueeze(0).unsqueeze(0)
        elif time_img.ndim == 3 and time_img.shape[0] == 1:  # Assume (1,H,W)
            time_img_kornia = time_img.unsqueeze(0)  # Make (1,1,H,W)
        elif (
            time_img.ndim == 4 and time_img.shape[0] == 1 and time_img.shape[1] == 1
        ):  # Already (1,1,H,W)
            time_img_kornia = time_img
        else:  # Fallback or error for unexpected shapes
            # This case should be handled based on expected time_img output
            # For now, assuming it becomes (1,1,H,W)
            if time_img.numel() == 0 or time_img.shape[-1] == 0 or time_img.shape[-2] == 0:
                return torch.tensor(float("inf"), device=C_prime_warped.device)  # Or a large error
            # Default attempt to reshape if not empty, might need specific handling
            time_img_kornia = time_img.view(1, 1, time_img.shape[-2], time_img.shape[-1])

        if (
            time_img_kornia.numel() == 0
            or time_img_kornia.shape[-1] == 0
            or time_img_kornia.shape[-2] == 0
        ):
            return torch.tensor(
                float("inf"), device=C_prime_warped.device
            )  # Or a very large number

        G = kornia.filters.spatial_gradient(time_img_kornia)  # Output (N,C,2,H,W), 2 is (dy,dx)
        Gy_component = G[:, :, 0, :, :]
        Gx_component = G[:, :, 1, :, :]
        return torch.sum(Gx_component**2 + Gy_component**2)

    def get_density(self, event_count_image_I: torch.Tensor) -> torch.Tensor:
        sum_total_events = torch.sum(event_count_image_I)
        # More direct way to count non-zero pixels for num_I
        num_pixels_with_events = torch.sum(event_count_image_I > 0).item()

        if num_pixels_with_events > 0:
            density = sum_total_events.float() / num_pixels_with_events
        else:
            density = torch.tensor(0.0, device=event_count_image_I.device, dtype=torch.float32)
        return density

    def optimize_motion_for_slice(
        self,
        current_event_slice_C: torch.Tensor,  # Raw events (x,y,t) for the slice
        initial_params: dict = None,
    ):
        """
        Optimizes the 4 motion parameters (hx, hy, hz, th) of self.model
        for the given current_event_slice_C using Algorithm 1 and Algorithm 2 from the paper.
        This method modifies self.model.parameters in place.
        """
        device = current_event_slice_C.device
        if initial_params:
            with torch.no_grad():
                self.model.hx.data = torch.tensor(
                    initial_params["hx"], device=device, dtype=torch.float32
                )
                self.model.hy.data = torch.tensor(
                    initial_params["hy"], device=device, dtype=torch.float32
                )
                self.model.hz.data = torch.tensor(
                    initial_params["hz"], device=device, dtype=torch.float32
                )
                self.model.th.data = torch.tensor(
                    initial_params["th"], device=device, dtype=torch.float32
                )
        # else: uses current parameters of self.model as M_{i-1}

        # --- Algorithm 1: Coarse Minimization on Time-Image T ---
        optimizer_alg1 = torch.optim.SGD(self.model.parameters(), lr=self.hparams.alg1_lr)

        for _ in range(self.hparams.alg1_iterations):
            params_before_update_alg1 = torch.stack(
                [p.clone().detach() for p in self.model.parameters()]
            )

            warped_C_prime_alg1 = self.model(
                current_event_slice_C
            )  # This is C' used for image generation
            count_img_alg1 = self.i_img(warped_C_prime_alg1)
            time_img_T = self.t_img(warped_C_prime_alg1, count_img_alg1)

            time_img_T_kornia = (
                time_img_T.view(1, 1, time_img_T.shape[-2], time_img_T.shape[-1])
                if time_img_T.ndim == 2
                else time_img_T
            )

            if (
                time_img_T_kornia.numel() == 0
                or time_img_T_kornia.shape[-1] == 0
                or time_img_T_kornia.shape[-2] == 0
            ):
                grad_hx, grad_hy, grad_hz, grad_th = torch.zeros(
                    4, device=device
                )  # No gradient if T is empty
            else:
                G_T = kornia.filters.spatial_gradient(time_img_T_kornia)
                Gy_T_component = G_T[:, :, 0, :, :].squeeze()  # dy
                Gx_T_component = G_T[:, :, 1, :, :].squeeze()  # dx

                num_active_pixels_T = torch.sum(count_img_alg1 > 0).float().clamp(min=1.0)

                grad_hx = torch.sum(Gx_T_component) / num_active_pixels_T
                grad_hy = torch.sum(Gy_T_component) / num_active_pixels_T

                H_T, W_T = time_img_T_kornia.shape[-2:]
                jj_coords, ii_coords = torch.meshgrid(
                    torch.arange(H_T, device=device, dtype=torch.float32),
                    torch.arange(W_T, device=device, dtype=torch.float32),
                    indexing="ij",
                )
                # Using uncentered coordinates as a direct interpretation
                grad_hz = (
                    torch.sum(Gx_T_component * ii_coords + Gy_T_component * jj_coords)
                    / num_active_pixels_T
                )
                grad_th = (
                    torch.sum(Gx_T_component * jj_coords - Gy_T_component * ii_coords)
                    / num_active_pixels_T
                )

            optimizer_alg1.zero_grad()
            with torch.no_grad():  # Manually assign computed gradients
                if self.model.hx.grad is None:
                    self.model.hx.grad = torch.zeros_like(self.model.hx)
                self.model.hx.grad.copy_(
                    grad_hx if torch.is_tensor(grad_hx) else torch.tensor(grad_hx, device=device)
                )
                if self.model.hy.grad is None:
                    self.model.hy.grad = torch.zeros_like(self.model.hy)
                self.model.hy.grad.copy_(
                    grad_hy if torch.is_tensor(grad_hy) else torch.tensor(grad_hy, device=device)
                )
                if self.model.hz.grad is None:
                    self.model.hz.grad = torch.zeros_like(self.model.hz)
                self.model.hz.grad.copy_(
                    grad_hz if torch.is_tensor(grad_hz) else torch.tensor(grad_hz, device=device)
                )
                if self.model.th.grad is None:
                    self.model.th.grad = torch.zeros_like(self.model.th)
                self.model.th.grad.copy_(
                    grad_th if torch.is_tensor(grad_th) else torch.tensor(grad_th, device=device)
                )
            optimizer_alg1.step()

            params_after_update_alg1 = torch.stack(
                [p.clone().detach() for p in self.model.parameters()]
            )
            param_change_norm_alg1 = torch.norm(
                params_after_update_alg1 - params_before_update_alg1
            )

            if param_change_norm_alg1 < self.hparams.alg1_xi:
                break

        # --- Algorithm 2: Fine Refinement on Event-Count Image I ---
        current_params = {name: p.item() for name, p in self.model.named_parameters()}

        warped_C_alg2_start = self.model(current_event_slice_C)
        count_img_alg2_current = self.i_img(warped_C_alg2_start)
        density_D_current = self.get_density(count_img_alg2_current)

        param_tensors_alg2 = [self.model.hx, self.model.hy, self.model.hz, self.model.th]
        param_names_alg2 = ["hx", "hy", "hz", "th"]

        for _ in range(self.hparams.alg2_iterations):
            density_before_param_sweep = density_D_current.clone()

            for p_idx, param_tensor in enumerate(param_tensors_alg2):
                original_param_val = param_tensor.item()

                perturb_val = self.hparams.alg2_perturb_abs
                if param_names_alg2[p_idx] in ["hz", "th"]:  # Potentially smaller for these
                    perturb_val *= 0.1

                # Test positive perturbation
                with torch.no_grad():
                    param_tensor.data += perturb_val
                density_pos = self.get_density(self.i_img(self.model(current_event_slice_C)))

                # Test negative perturbation
                with torch.no_grad():
                    param_tensor.data -= 2 * perturb_val
                density_neg = self.get_density(self.i_img(self.model(current_event_slice_C)))

                with torch.no_grad():
                    param_tensor.data = torch.tensor(
                        original_param_val, device=device, dtype=torch.float32
                    )

                if density_pos > density_D_current and density_pos >= density_neg:
                    with torch.no_grad():
                        param_tensor.data += perturb_val
                    density_D_current = density_pos
                elif density_neg > density_D_current:
                    with torch.no_grad():
                        param_tensor.data -= perturb_val
                    density_D_current = density_neg

            density_change_abs = torch.abs(density_D_current - density_before_param_sweep)
            if density_before_param_sweep.abs() > 1e-9:
                density_change_rel = density_change_abs / density_before_param_sweep.abs()
                if density_change_rel < self.hparams.alg2_xi:
                    break
            elif density_change_abs < self.hparams.alg2_xi:
                break  # Absolute for small densities

        optimized_params = {name: p.item() for name, p in self.model.named_parameters()}
        return optimized_params


# Example usage (not part of the class, for testing)
if __name__ == "__main__":
    # Test WarpTransform
    print("--- Testing WarpTransform ---")
    warp_model_test = WarpTransform()
    # Ensure parameters are on CUDA if testing with CUDA events
    # warp_model_test.to('cuda')
    # sample_events_xyt = torch.randn([5000, 3], device='cuda') * torch.tensor([239, 179, 0.1], device='cuda') # Example events
    sample_events_xyt = torch.rand([5000, 3]) * torch.tensor(
        [239.0, 179.0, 0.1]
    )  # x, y within DVS range, t small
    sample_events_xyt[:, 2] = torch.sort(sample_events_xyt[:, 2])[
        0
    ]  # ensure timestamps are sorted for a slice

    print("Sample input events shape:", sample_events_xyt.shape)
    warped_output = warp_model_test(sample_events_xyt)
    print("Warped output shape:", warped_output.shape)
    print("Warped output sample:", warped_output[0])

    print("\n--- Testing LitWarpTransform Optimization ---")
    # Assuming dvs_native_width=240, dvs_native_height=180 for the sample_events_xyt
    lit_warp_optimizer = LitWarpTransform(
        dvs_native_width=240, dvs_native_height=180, unit_pixel_size=1.0
    )
    # lit_warp_optimizer.to('cuda') # If using CUDA

    # Example: Reset parameters for a fresh optimization
    initial_guess = {"hx": 0.0, "hy": 0.0, "hz": 0.0, "th": 0.0}

    print(
        f"Initial model params: hx={lit_warp_optimizer.model.hx.item():.4f}, hy={lit_warp_optimizer.model.hy.item():.4f}, hz={lit_warp_optimizer.model.hz.item():.4f}, th={lit_warp_optimizer.model.th.item():.4f}"
    )

    optimized_motion = lit_warp_optimizer.optimize_motion_for_slice(
        sample_events_xyt, initial_params=initial_guess
    )

    print(
        f"Optimized model params: hx={optimized_motion['hx']:.4f}, hy={optimized_motion['hy']:.4f}, hz={optimized_motion['hz']:.4f}, th={optimized_motion['th']:.4f}"
    )
    print(
        f"Final internal model params: hx={lit_warp_optimizer.model.hx.item():.4f}, hy={lit_warp_optimizer.model.hy.item():.4f}, hz={lit_warp_optimizer.model.hz.item():.4f}, th={lit_warp_optimizer.model.th.item():.4f}"
    )
