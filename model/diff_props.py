""" Differential geometry properties. Implements the computation of the 1st and
2nd order differential quantities of the 3D points given a UV coordinates and a
mapping f: R^{2} -> R^{3}, which takes a UV 2D point and maps it to a xyz 3D
point. The differential quantities are computed using analytical formulas
involving derivatives d_f/d_uv which are practically computed using Torch's
autograd mechanism. The computation graph is still built and it is possible to
backprop through the diff. quantities computation. The computed per-point
quantities are the following: normals, mean curvature, gauss. curvature.

Author: Jan Bednarik, jan.bednarik@epfl.ch
Date: 7.2.2020
"""

# 3rd party
import torch
import torch.nn as nn
import torch.autograd as ag
import torch.nn.functional as F



class DiffGeomProps(nn.Module):
    """ Computes the differential geometry properties including normals,
    mean curvature, gaussian curvature, first fundamental form.

    Args:
        normals (bool): Whether to compute normals.
        curv_mean (bool): Whether to compute mean curvature.
        curv_gauss (bool): Whether to compute gaussian curvature.
        fff (bool): Whether to compute first fundamental form.
        gpu (bool): Whether to use GPU.
    """
    def __init__(self, device, normals=True, curv_mean=True, curv_gauss=True, fff=False,
                 gpu=True):
        nn.Module.__init__(self)
        self.device = device
        self._comp_normals = normals
        self._comp_cmean = curv_mean
        self._comp_cgauss = curv_gauss
        self._comp_fff = fff

    def forward(self, xyz, uv):
        """ Computes the 1st and 2nd order derivative quantities, namely
        normals, mean curvature, gaussian curvature, first fundamental form.

        Args:
            xyz (torch.Tensor): 3D points, output 3D space (B, Num_Prim, 3, Num_Points).
            uv (torch.Tensor): 2D points, parameter space, shape (B, M, 2).

        Returns:
            dict: Depending on `normals`, `curv_mean`, `curv_gauss`, `fff`
                includes normals, mean curvature, gauss. curvature and first
                fundamental form as torch.Tensor.
        """

        # Return values.
        ret = {}

        if not (self._comp_normals or self._comp_cmean or self._comp_cgauss or
                self._comp_fff):
            return ret

        # Data shape.
        B, P, _, M = xyz.shape

        # 1st order derivatives d_fx/d_uv, d_fy/d_uv, d_fz/d_uv.
        dxyz_duv = []
        for o in range(3):
            derivs = self.df(xyz[:, :, o], uv)  # (B, M, 2)
            assert(derivs.shape == (B, 2, M*P))
            dxyz_duv.append(derivs.transpose(1,2).contiguous().unsqueeze(2))

        # Jacobian, d_xyz / d_uv.
        J_f_uv = torch.cat(dxyz_duv, dim=2) # B, M, 3, 2

        # normals
        normals = F.normalize(
            torch.cross(J_f_uv[..., 0],
                        J_f_uv[..., 1], dim=2), p=2, dim=2)  # (B, P*M, 3)
        assert (normals.shape == (B, P*M, 3))

        # Save normals.
        if self._comp_normals:
            ret['normals'] = normals

        if self._comp_fff or self._comp_cmean or self._comp_cgauss:
            # 1st fundamental form (g)
            g = torch.matmul(J_f_uv.transpose(2, 3), J_f_uv)
            assert (g.shape == (B, P*M, 2, 2))

            # Save first fundamental form, only E, F, G terms, instead of
            # the whole matrix [E F; F G].
            if self._comp_fff:
                ret['fff'] = g.reshape((B, P*M, 4))[:, :, [0, 1, 3]]  # (B, P*M, 3)

        if self._comp_cmean or self._comp_cgauss:
            # determinant of g.
            detg = g[:, :, 0, 0] * g[:, :, 1, 1] - g[:, :, 0, 1] * g[:, :, 1, 0]
            assert (detg.shape == (B, P*M))

            # 2nd order derivatives, d^2f/du^2, d^2f/dudv, d^2f/dv^2
            d2xyz_duv2 = []
            for o in range(3):
                for i in range(2):
                    deriv = self.df(dxyz_duv[o][:, :, :, i], uv)  # (B, M, 2)
                    assert(deriv.shape == (B, 2, P*M))
                    d2xyz_duv2.append(deriv.transpose(1,2).contiguous())

            d2xyz_du2 = torch.stack(
                [d2xyz_duv2[0][..., 0], d2xyz_duv2[2][..., 0],
                 d2xyz_duv2[4][..., 0]], dim=2)  # (B, M, 3)
            d2xyz_dudv = torch.stack(
                [d2xyz_duv2[0][..., 1], d2xyz_duv2[2][..., 1],
                 d2xyz_duv2[4][..., 1]], dim=2)  # (B, M, 3)
            d2xyz_dv2 = torch.stack(
                [d2xyz_duv2[1][..., 1], d2xyz_duv2[3][..., 1],
                 d2xyz_duv2[5][..., 1]], dim=2)  # (B, M, 3)
            assert(d2xyz_du2.shape == (B, P*M, 3))
            assert(d2xyz_dudv.shape == (B, P*M, 3))
            assert(d2xyz_dv2.shape == (B, P*M, 3))

            # Each (B, M)
            gE, gF, _, gG = g.reshape((B, P*M, 4)).permute(2, 0, 1)
            assert (gE.shape == (B, P*M))

        # Compute mean curvature.
        if self._comp_cmean:
            cmean = torch.sum((-normals / detg[..., None]) *
                (d2xyz_du2 * gG[..., None] - 2. * d2xyz_dudv * gF[..., None] +
                 d2xyz_dv2 * gE[..., None]), dim=2) * 0.5
            ret['cmean'] = cmean

        # Compute gaussian curvature.
        if self._comp_cgauss:
            iiL = torch.sum(d2xyz_du2 * normals, dim=2)
            iiM = torch.sum(d2xyz_dudv * normals, dim=2)
            iiN = torch.sum(d2xyz_dv2 * normals, dim=2)
            cgauss = (iiL * iiN - iiM.pow(2)) / (gE * gF - gG.pow(2))
            ret['cgauss'] = cgauss

        return ret

    def df(self, x, wrt):
        B, P, M = x.shape
        grads =  ag.grad(x.flatten(), wrt,
                       grad_outputs=torch.ones(B * P * M, dtype=torch.float32).
                       to(self.device), create_graph=True)
        return torch.cat(grads, 2)
