import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeDiversityRegularizer(nn.Module):
    """Encourages class-wise prototype diversity over spatial regions.

    The implementation mirrors the formulation in the paper: for each
    class :math:`c`, we approximate the set :math:`\Omega_c` of spatial
    locations predicted as class :math:`c`, and penalize correlations
    between the regional activation distributions :math:`v_{c,u}` of the
    class prototypes.
    """

    def __init__(self,
                 num_prototypes_per_class: int,
                 omega_window: int = 7,
                 omega_min_mass: float = 0.05,
                 debug: bool = False,
                 debug_every: int = 200):
        super().__init__()
        if omega_window < 1 or omega_window % 2 == 0:
            raise ValueError("omega_window must be a positive odd integer")
        if debug_every < 1:
            raise ValueError("debug_every must be a positive integer")
        self.num_prototypes_per_class = num_prototypes_per_class
        self.omega_window = omega_window
        self.omega_min_mass = omega_min_mass
        self.debug = debug
        self.debug_every = debug_every
        self._call_count = 0
        self.last_debug = None

    def forward(self, feature_map, prototypes, pseudo_mask):
        """Compute :math:`\mathcal{L}_{\text{div}}`.

        Args:
            feature_map: Stage-4 feature tensor :math:`F \in \mathbb{R}^{B\times D\times H\times W}`.
            prototypes:  Learnable prototypes :math:`P \in \mathbb{R}^{(Ck)\times D}` aligned with ``feature_map``.
            pseudo_mask: Pseudo labels :math:`\Omega` with shape ``[B, H, W]`` (0 background, 1..C foreground).
        """
        B, D, H, W = feature_map.shape

        if prototypes.shape[1] != D:
            raise ValueError(
                f"Prototype dim {prototypes.shape[1]} != feature dim {D}. "
                "Pass stage-4 projected prototypes instead."
            )

        total = feature_map.new_tensor(0.0)
        eps = 1e-8
        num_classes = prototypes.shape[0] // self.num_prototypes_per_class

        # Debug statistics accumulators
        active_classes = 0
        active_images = 0
        sum_valid_locations = 0.0
        sum_mask_mass = 0.0
        sum_abs_corr = 0.0
        max_abs_corr = 0.0
        sum_loss_values = 0.0

        for b in range(B):
            class_losses = []
            Fb = F.normalize(feature_map[b:b + 1], p=2, dim=1)
            omega_b = pseudo_mask[b]

            for c in range(num_classes):
                omega_c = (omega_b == (c + 1)).float().view(1, 1, H, W)
                if omega_c.sum() < 1:
                    continue

                start = c * self.num_prototypes_per_class
                end = (c + 1) * self.num_prototypes_per_class
                Pc = F.normalize(prototypes[start:end], p=2, dim=1)
                K = Pc.shape[0]
                if K < 2:
                    continue

                weight = Pc.view(K, D, 1, 1)
                sim = F.conv2d(Fb, weight)

                window = self.omega_window
                num = F.avg_pool2d(sim * omega_c, kernel_size=window, stride=1, padding=window // 2)
                den = F.avg_pool2d(omega_c, kernel_size=window, stride=1, padding=window // 2)
                region_sim = num / (den + eps)

                den_flat = den.view(-1)
                valid = den_flat > self.omega_min_mass
                valid_count = int(valid.sum().item())
                if valid_count < 2:
                    continue

                a = region_sim.view(K, -1).t()
                a = a[valid]
                a = a - a.mean(dim=0, keepdim=True)
                a = a / (a.norm(dim=0, keepdim=True) + eps)

                gram = (a.t() @ a) / a.shape[0]
                off_diag = gram - torch.diag_embed(gram.diagonal())
                denom = K * (K - 1)
                loss_c = off_diag.pow(2).sum() / denom
                class_losses.append(loss_c)

                # Debug stats (detach to avoid gradient tracking)
                active_classes += 1
                sum_valid_locations += valid_count
                mask_mass = float(den_flat[valid].mean().item()) if valid_count > 0 else 0.0
                sum_mask_mass += mask_mass
                abs_corr_mean = float(off_diag.abs().sum().detach().item() / denom)
                sum_abs_corr += abs_corr_mean
                max_abs_corr = max(max_abs_corr, float(off_diag.abs().max().detach().item()))
                sum_loss_values += float(loss_c.detach().item())

            if class_losses:
                total = total + torch.stack(class_losses).mean()
                active_images += 1

        loss = total / B

        # Store debug snapshot and optionally print
        denom_classes = max(active_classes, 1)
        debug_info = {
            "active_images": active_images,
            "active_classes": active_classes,
            "avg_valid_locations": sum_valid_locations / denom_classes,
            "avg_mask_mass": sum_mask_mass / denom_classes,
            "avg_abs_corr": sum_abs_corr / denom_classes,
            "max_abs_corr": max_abs_corr,
            "avg_loss_per_class": sum_loss_values / denom_classes,
        }
        self.last_debug = debug_info

        self._call_count += 1
        if self.debug and self._call_count % self.debug_every == 0:
            print(f"[PrototypeDiversityRegularizer] step={self._call_count} "
                  f"info={debug_info}")

        return loss
