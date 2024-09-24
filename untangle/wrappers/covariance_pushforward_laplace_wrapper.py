"""Low-rank Laplace approximation with covariance pushfwd."""

import logging
import re
import time
from dataclasses import dataclass
from math import sqrt
from warnings import warn

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, eigsh
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn.utils.convert_parameters import parameters_to_vector

from untangle.utils.derivative import jvp, vjp
from untangle.utils.metric import calibration_error
from untangle.wrappers.model_wrapper import DistributionalWrapper

logger = logging.getLogger(__name__)


@dataclass
class QuadRootTerms:
    """Container of factors for the covariance matrix `Sigma` from Laplace.

    Either `d` or `lambda_k` must be provided. When `lambda_k` is
    provided, the terms correspond to the factorization
    `sqrt(Sigma) = mul * U_k diag(lambda_k)`, which is a `p x k` matrix. When `d` is
    provided, the factorization `sqrt(Sigma) = mul * I + U_k diag(d) U_k^T` is used,
    which is a `p x p` matrix.
    """

    U_k: torch.Tensor
    mul: float
    d: torch.Tensor = None
    lambda_k: torch.Tensor = None


class Quadratic:
    """A quadratic function.

    q(theta) = 0.5 * (theta - theta_0)^T * B_0 * (theta - theta_0)
               (theta - theta_0)^T * g_0
               + c_0
    """

    def __init__(self, B_0v, g_0, c_0, theta_0):
        # Save args
        self.B_0v = B_0v
        self.g_0 = g_0
        self.c_0 = float(c_0)
        self.theta_0 = theta_0
        self.D = len(theta_0)
        self.device = theta_0.device
        self.dtype = theta_0.dtype

        # Sanity check
        if g_0.shape != theta_0.shape:
            raise ValueError

    def q(self, theta):
        """The value of the quadratic model q at `theta`."""
        dtheta = theta - self.theta_0
        q_val = self.c_0
        q_val += torch.inner(self.g_0, dtheta)
        q_val += 0.5 * torch.inner(dtheta, self.B_0v(dtheta))
        return q_val.item()

    def nabla_q(self, theta):
        """The gradient of q at `theta`."""
        return self.B_0v(theta - self.theta_0) + self.g_0

    def nabla2_q(self, theta, vec):
        """Matrix vector product with q's Hessian at `theta`."""
        del theta
        return self.B_0v(vec)

    @staticmethod
    def check_direction(d):
        """Check if direction is normalized. If not, raise warning."""
        d_norm = torch.linalg.vector_norm(d).item()
        if not np.isclose(1.0, d_norm, rtol=1e-3):
            warn(f"Direction with norm {d_norm} (should be 1.0).", stacklevel=1)

    def get_scipy_linop(self):
        """Return a `scipy` `LinearOperator` for the curvature matrix."""

        def matvec(vec):
            """Matrix vector product with the curvature matrix.

            Note that `nabla2_q` is independent from `theta`
            """
            vec_torch = torch.from_numpy(vec).type(self.dtype).to(self.device)
            mv = self.nabla2_q(theta=None, vec=vec_torch)
            return mv.detach().cpu().numpy()

        return LinearOperator((self.D, self.D), matvec=matvec)

    def get_eigendecomposition(self, k):
        """Return the eigenvalues and eigenvectors of the curvature matrix."""
        # Use largest eigenvalues of the quadratic's Hessian
        # Extract curvature linear operator and approximate eigenvectors
        curvop = self.get_scipy_linop()
        eigvals, eigvecs = eigsh(curvop, k=k)

        # Convert to PyTorch tensors and reverse order
        eigvals = torch.from_numpy(eigvals).to(self.device)
        eigvecs = torch.from_numpy(eigvecs).to(self.device)

        # Sort by eigenvalues
        _, sort_idx = torch.sort(eigvals, descending=True)
        return eigvals[sort_idx], eigvecs[:, sort_idx]


class CovariancePushforwardLaplaceWrapper(DistributionalWrapper):
    """Low-rank Laplace that pushes forward the weight-space covariance matrix."""

    def __init__(
        self,
        model,
        loss_function,
        rank,
        predictive_fn,
        mask_regex,
        *,
        use_eigval_prior=False,
    ):
        super().__init__(model)
        # Save args
        self.rank = rank
        self.loss_function = loss_function
        self.use_eigval_prior = use_eigval_prior
        self.predictive_fn = predictive_fn
        self._loss_and_grad = None

        if mask_regex is not None:
            self.apply_parameter_mask(mask_regex=mask_regex)

        # Model parameters
        self.theta_0_list = [
            param for param in model.parameters() if param.requires_grad
        ]
        self.theta_0_vec = parameters_to_vector(self.theta_0_list).detach()
        self.prior_precision = 1.0
        self.quadratic = None
        self.quad_root_terms = None
        self._regularizer_diag_hessian_0 = None

    def apply_parameter_mask(self, mask_regex):
        if mask_regex is not None:
            for param_name, param in self.model.named_parameters():
                if re.match(mask_regex, param_name):
                    param.requires_grad = False

    def perform_laplace_approximation(self, train_loader, val_loader):
        self.train_loader = train_loader
        # Construct quadratic model
        self.quadratic = Quadratic(
            B_0v=self.get_B_0v,
            g_0=self.get_g_0(),
            c_0=self.get_c_0(),
            theta_0=self.theta_0_vec,
        )

        self.quad_root_terms = self.get_quad_root_terms(
            len(self.train_loader.dataset), rank_k=self.rank
        )
        self.prior_precision = self._optimize_prior_precision_cv(val_loader)

    @property
    def prior_precision(self):
        return self._prior_precision

    @prior_precision.setter
    def prior_precision(self, value):
        self._regularizer_diag_hessian_0 = None
        self._prior_precision = value

    @staticmethod
    def _get_ece(out_dist, targets):
        confidences, predictions = out_dist.max(dim=-1)  # [B]
        correctnesses = predictions.eq(targets).int()

        return calibration_error(
            confidences=confidences, correctnesses=correctnesses, num_bins=15, norm="l1"
        )

    def _optimize_prior_precision_cv(
        self,
        val_loader,
        log_prior_prec_min=-1,
        log_prior_prec_max=2,
        grid_size=50,
    ):
        interval = torch.logspace(log_prior_prec_min, log_prior_prec_max, grid_size)
        self.prior_precision = self._gridsearch(
            interval=interval,
            val_loader=val_loader,
        )

        logger.info(f"Optimized prior precision is {self.prior_precision}.")

    def _gridsearch(
        self,
        interval,
        val_loader,
    ):
        results = []
        prior_precs = []
        for prior_prec in interval:
            logger.info(f"Trying {prior_prec}...")
            start_time = time.perf_counter()
            self.prior_precision = prior_prec

            try:
                out_dist, targets = self._validate(
                    val_loader=val_loader,
                )
                result = self._get_ece(out_dist, targets).item()
                accuracy = out_dist.argmax(dim=-1).eq(targets).float().mean()
            except RuntimeError as error:
                logger.info(f"Caught an exception in validate: {error}")
                result = float("inf")
                accuracy = float("NaN")
            logger.info(
                f"Took {time.perf_counter() - start_time} seconds, result: {result}, "
                f"accuracy {accuracy}"
            )
            results.append(result)
            prior_precs.append(prior_prec)

        return prior_precs[np.argmin(results)]

    @torch.no_grad()
    def _validate(self, val_loader):
        self.model.eval()
        device = next(self.model.parameters()).device
        output_means = []
        targets = []

        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)

            mean, var = self(input)
            out = self.predictive_fn(mean, var)

            output_means.append(out)
            targets.append(target)

        return torch.cat(output_means, dim=0), torch.cat(targets, dim=0)

    def get_quad_root_terms(
        self,
        N_train,
        rank_k,
    ):
        """Compute the factors of the root `V` of `Sigma` from Laplace.

        `Sigma = V^T V` (approximately, depending on the
        rank `rank_k`). If `quad_dd2` is given, keep the eigenvectors of `quad` but
        replace the eigenvalues with directional curvatures computed on `quad_dd2`.
        """
        logger.info("Computing the eigencomposition of the Hessian...")
        eigvals_k, U_k = self.quadratic.get_eigendecomposition(k=rank_k)
        logger.info(
            f"eigvals in [{eigvals_k.min().item():.6e}, {eigvals_k.max().item():.6e}]"
        )

        # Check for small eigenvalues/directional curvatures
        if eigvals_k.min().item() <= 1e-5:
            warn(
                "Warning: Small eigenvalues might lead to instabilities.", stacklevel=1
            )

        if self.use_eigval_prior:
            # Calculate low-rank terms including the prior
            mul = 1 / sqrt(N_train * self.prior_precision)
            d = 1 / torch.sqrt(N_train * eigvals_k) - mul
            return QuadRootTerms(U_k=U_k, d=d, mul=mul)

        mul = 1 / sqrt(N_train)
        lambda_k = 1 / torch.sqrt(eigvals_k)
        return QuadRootTerms(U_k=U_k, lambda_k=lambda_k, mul=mul)

    @staticmethod
    def create_basis_tensor(batch_size, vector_size, i):
        # Create the i-th canonical basis vector
        e_i = torch.zeros(vector_size)
        e_i[i] = 1.0

        # Create a 3D tensor with the basis vector on the diagonal
        result = torch.zeros(batch_size, vector_size, vector_size)
        result[:, range(vector_size), range(vector_size)] = e_i

        return result

    def forward(self, x):
        param_list = self.theta_0_list

        # w ~ N(0, 1)
        # mul * w + U_K @ D @ U_K^T @ w ~ N(0, AA^T)
        # where A = mul * I + U_K @ D @ U_K^T
        # => S = AA^T = (mul * I + U_K @ D @ U_K^T) @ (mul * I + U_K @ D @ U_K^T)
        #             = mul**2 * I + 2 * mul * U_K @ D @ U_K^T + U_K @ D^2 @ U_K^T
        # => JSJ^T = mul**2 * JJ^T {1}
        #            + 2 * mul * (J @ U_K @ sqrt(D)) (J @ U_K @ sqrt(D))^T {2}
        #            + (J @ U_K @ D) (J @ U_K @ D)^T {3}

        with torch.enable_grad():
            y = self.model(x)  # [B, C]

            if self.use_eigval_prior:
                a = torch.zeros(
                    (y.shape[0], self.model.num_classes), dtype=y.dtype, device=y.device
                )
                for b in range(y.shape[0]):
                    y_b = y[b]
                    for i in range(self.model.num_classes):
                        e_bi = torch.zeros_like(y_b)
                        e_bi[i] = 1.0
                        J_bi = vjp(y=y_b, x=param_list, v=e_bi, retain_graph=True)
                        a[b, i] = self.quad_root_terms.mul**2 * self.square_sum(J_bi)

                sqrt_abs_d = self.quad_root_terms.d.abs().sqrt()
                sign_d = self.quad_root_terms.d.sign()
                u = vector_to_parameter_list(
                    (sign_d * sqrt_abs_d).unsqueeze(dim=1) * self.quad_root_terms.U_k.T,
                    self.theta_0_list,
                )

                B = jvp(
                    y=y, x=param_list, v=u, is_grads_batched=True, retain_graph=True
                )[0]
                b = 2 * self.quad_root_terms.mul * B.square().sum(dim=0)

                v = vector_to_parameter_list(
                    self.quad_root_terms.d.unsqueeze(dim=1)
                    * self.quad_root_terms.U_k.T,
                    self.theta_0_list,
                )
                C = jvp(y=y, x=param_list, v=v, is_grads_batched=True)[0]
                c = self.quad_root_terms.mul * C.square().sum(dim=0)

                return y, a + b + c

            u = vector_to_parameter_list(
                self.quad_root_terms.mul
                * self.quad_root_terms.lambda_k.unsqueeze(dim=1)
                * self.quad_root_terms.U_k.T,
                self.theta_0_list,
            )
            A = jvp(y=y, x=param_list, v=u, is_grads_batched=True, retain_graph=True)[0]
            a = A.square().sum(dim=0)

            return y, a

    @staticmethod
    def vector_to_parameter_list(vec, parameters):
        """Convert the vector `vec` to a parameter-list format matching `parameters`.

        This function is the inverse of `parameters_to_vector` from
        `torch.nn.utils`. In contrast to `vector_to_parameters`, which replaces the
        value of the parameters, this function leaves the parameters unchanged and
        returns a list of parameter views of the vector.

        Args:
            vec (torch.Tensor): The vector representing the parameters. This vector
                is converted to a parameter-list format matching `parameters`.
            parameters (iterable): An iterable of `torch.Tensor`s containing the
                parameters. These parameters are not changed by the function.

        Raises:
            Warning if not all entries of `vec` are converted.
        """
        if not isinstance(vec, torch.Tensor):
            msg = f"`vec` should be a torch.Tensor, not {type(vec)}"
            raise TypeError(msg)

        # Put slices of `vec` into `params_list`
        params_list = []
        pointer = 0
        ndim = vec.ndim()
        for param in parameters:
            num_param = param.numel()
            params_list.append(
                vec[pointer : pointer + num_param].view_as(param).data
                if ndim == 1
                else vec[:, pointer : pointer + num_param].view(-1, *param.shape)
            )
            pointer += num_param

        # Make sure all entries of the vector have been used (i.e. that `vec` and
        # `parameters` have the same number of elements)
        if pointer != len(vec):
            warn("Not all entries of `vec` have been used.", stacklevel=1)

        return params_list

    def calc_l2_loss_0(self):
        layer_norms = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if param.ndim > 1 and not name.endswith(".bias"):
                layer_norms.append(self.prior_precision * param.square().sum())
        l2_loss_0 = 0.5 * sum(layer_norms)

        return l2_loss_0

    @staticmethod
    def square_sum(a_list):
        res = 0.0
        for a_param in a_list:
            res += a_param.square().sum()

        return res

    def get_loss_and_grad(self):
        """Computes the loss and gradient at `theta_0`."""
        if self._loss_and_grad is not None:
            return self._loss_and_grad

        # Initialize loss and gradient
        loss = 0.0
        device = next(self.model.parameters()).device
        grad = torch.zeros_like(self.theta_0_vec).to(device)

        # Accumulate
        num_data = 0

        # Loop over mini-batches of data
        for inputs, targets in self.train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            with torch.enable_grad():
                outputs = self.model(inputs)
                mb_size = inputs.shape[0]
                num_data += mb_size

                # Update loss
                mb_loss = self.loss_function(outputs, targets)
                loss += mb_size * mb_loss

                # Update gradient
                mb_grad = torch.autograd.grad(mb_loss, self.theta_0_list)
                grad += mb_size * parameters_to_vector(mb_grad).detach()

        # Re-scale loss and gradient
        loss = loss / num_data
        grad = grad / num_data

        with torch.enable_grad():
            # Add L2 regularizer to loss and gradient
            l2_loss_0 = self.calc_l2_loss_0()
            loss = loss + l2_loss_0

            # We have to include all parameters in the compute graph
            l2_grad = torch.autograd.grad(
                l2_loss_0 + 0.0 * sum(p.sum() for p in self.theta_0_list),
                self.theta_0_list,
                allow_unused=True,
            )
        grad = grad + parameters_to_vector(l2_grad).detach()

        self._loss_and_grad = loss, grad
        return self._loss_and_grad

    def get_c_0(self):
        """Constant term in the quadratic.

        The constant term in the quadratic model is the loss at `theta_0` plus the
        L2 regularizer at `theta_0`.
        """
        return self.get_loss_and_grad()[0]

    def get_g_0(self):
        """Gradient at `theta_0`.

        The gradient at `theta_0` is the gradient of the loss at `theta_0` plus the
        gradient of the L2 regularizer at `theta_0`.
        """
        return self.get_loss_and_grad()[1]

    def regularizer_diag_hessian_0(self):
        """The diagonal of the L2 regularizer's Hessian.

        The L2 regularizer is given by
        `0.5 * l2_strength * ||theta_0||^2`, i.e. its Hessian is diagonal and the
        entries are all `self.l2_strength`. However, it might be that not all parameters
        in the network (e.g. the weights but not the biases) are regularized. This is
        automatically taken care of here: In order to find out which parameters are
        regularized, we compute the gradient of the L2 loss w.r.t. the parameters and
        check which gradients are not `None`. These are replaced with
        `self.prior_precision` and the others with `0.0` (each with the correct shape).
        """
        if self._regularizer_diag_hessian_0 is not None:
            return self._regularizer_diag_hessian_0

        with torch.enable_grad():
            l2_grad = torch.autograd.grad(
                self.calc_l2_loss_0(),
                self.theta_0_list,
                allow_unused=True,
            )

        hess_diag = []
        for p, g in zip(self.theta_0_list, l2_grad, strict=True):
            if g is None:
                # `p` does not appear in l2-loss --> entries in Hessian are zero
                hess_diag.append(torch.zeros_like(p))
            else:
                # `p` does appear in l2-loss --> entries in Hessian are
                # `prior_precision`
                hess_diag.append(self.prior_precision * torch.ones_like(p))

        hess_diag = parameters_to_vector(hess_diag)

        # Sanity check and return diagonal of the Hessian
        if hess_diag.shape != self.theta_0_vec.shape:
            raise ValueError

        self._regularizer_diag_hessian_0 = hess_diag
        return self._regularizer_diag_hessian_0

    def get_B_0v(self, vec):
        """Matrix vector product with the curvature matrix at `theta_0`.

        The curvature matrix has three terms: (i) the curvature of the mini-batch loss
        `self.loss_0`, (ii) the Hessian of the L2 loss of the model weights and (iii) a
        damping term.
        """
        # Convert vector to parameter list
        if vec.shape != self.theta_0_vec.shape:
            raise ValueError

        v_list = vector_to_parameter_list(vec, self.theta_0_list)

        # ------------------------------------------------------------------------------
        # (i) Vector product with GGN/Hessian of mini-batch loss
        # ------------------------------------------------------------------------------

        # Initialize Hv
        device = next(self.model.parameters()).device
        B_0v = torch.zeros_like(self.theta_0_vec).to(device)

        # Accumulate
        num_data = 0
        # Loop over mini-batches of data
        for inputs, targets in self.train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Compute GGN vector product
            with torch.enable_grad():
                # Forward pass
                outputs = self.model(inputs)
                mb_size = inputs.shape[0]
                num_data += mb_size
                mb_loss = self.loss_function(outputs, targets)
                Hv = ggn_vector_product_from_plist(
                    mb_loss,
                    outputs,
                    self.theta_0_list,
                    v_list,
                )
            B_0v += mb_size * parameters_to_vector(Hv).detach()

        # Re-scale
        B_0v = B_0v / num_data

        # ------------------------------------------------------------------------------
        # (ii) Vector product with Hessian of L2 loss
        # ------------------------------------------------------------------------------
        B_0v += self.regularizer_diag_hessian_0() * vec  # element-wise product!

        return B_0v


def ggn_vector_product_from_plist(
    loss: Tensor, output: Tensor, plist: list[Parameter], v: list[Tensor]
) -> tuple[Tensor]:
    """Multiply a vector with a sub-block of the generalized Gauss-Newton/Fisher.

    Args:
        loss: Scalar tensor that represents the loss.
        output: Model output.
        plist: list of trainable parameters whose GGN block is used for multiplication.
        v: Vector specified as list of tensors matching the sizes of ``plist``.

    Returns:
        GGN-vector product in list format, i.e. as list that matches the sizes of
        ``plist``.
    """
    Jv = R_op(output, plist, v, retain_graph=True)
    HJv = hvp(loss, output, Jv, retain_graph=True)
    JTHJv = L_op(output, plist, HJv, retain_graph=False)
    return JTHJv


def R_op(ys, xs, vs, *, retain_graph=True, detach=True):
    """Multiplies the vector `vs` with the Jacobian of `ys` w.r.t. `xs`."""
    if isinstance(ys, tuple):
        ws = [torch.zeros_like(y).requires_grad_(True) for y in ys]
    else:
        ws = torch.zeros_like(ys).requires_grad_(True)

    gs = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )

    re = torch.autograd.grad(
        gs, ws, grad_outputs=vs, create_graph=True, retain_graph=True, allow_unused=True
    )

    if detach:
        return tuple(j.detach() for j in re)

    return re


def L_op(ys, xs, ws, *, retain_graph=True, detach=True):
    """Multiplies the vector `ws` with the transposed Jacobian of `ys` w.r.t. `xs`."""
    vJ = torch.autograd.grad(
        ys,
        xs,
        grad_outputs=ws,
        create_graph=True,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    if detach:
        return tuple(j.detach() for j in vJ)

    return vJ


def hvp(y, params, v, grad_params=None, *, retain_graph=True, detach=True):
    """Multiplies the vector `v` with the Hessian, `v = H @ v`.

    `H` is the Hessian of `y` w.r.t. `params`.

    Example usage:
    ```
    X, Y = data()
    model = torch.nn.Linear(784, 10)
    lossfunc = torch.nn.CrossEntropyLoss()

    loss = lossfunc(output, Y)

    v = list([torch.randn_like(p) for p in model.parameters])

    Hv = hessian_vector_product(loss, list(model.parameters()), v)
    ```

    Parameters:
    -----------
        y: torch.Tensor
        params: torch.Tensor or [torch.Tensor]
        v: torch.Tensor or [torch.Tensor]
            Shapes must match `params`
        grad_params: torch.Tensor or [torch.Tensor], optional
            Gradient of `y` w.r.t. `params`. If the gradients have already
            been computed elsewhere, the first of two backpropagations can
            be saved. `grad_params` must have been computed with
            `create_graph = True` to not destroy the computation graph for
            the second backward pass.
        detach: bool, optional
            Whether to detach the output from the computation graph
            (default: True)
    """
    if grad_params is not None:
        df_dx = tuple(grad_params)
    else:
        df_dx = torch.autograd.grad(
            y, params, create_graph=True, retain_graph=retain_graph
        )

    Hv = R_op(df_dx, params, v)

    if detach:
        return tuple(j.detach() for j in Hv)

    return Hv


def vector_to_parameter_list(vec, parameters):
    """Convert the vector `vec` to a parameter-list format matching `parameters`.

    This function is the inverse of `parameters_to_vector` from
    `torch.nn.utils`. In contrast to `vector_to_parameters`, which replaces the
    value of the parameters, this function leaves the parameters unchanged and
    returns a list of parameter views of the vector.

    Args:
        vec (torch.Tensor): The vector representing the parameters. This vector
            is converted to a parameter-list format matching `parameters`.
        parameters (iterable): An iterable of `torch.Tensor`s containing the
            parameters. These parameters are not changed by the function.

    Raises:
        Warning if not all entries of `vec` are converted.
    """
    if not isinstance(vec, torch.Tensor):
        msg = f"`vec` should be a torch.Tensor, not {type(vec)}"
        raise TypeError(msg)

    is_batched = vec.ndim == 2

    # Put slices of `vec` into `params_list`
    params_list = []
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param_shape = (vec.shape[0], *param.shape) if is_batched else param.shape
        params_list.append(
            vec[..., pointer : pointer + num_param].view(param_shape).data
        )
        pointer += num_param

    # Make sure all entries of the vector have been used (i.e. that `vec` and
    # `parameters` have the same number of elements)
    if pointer != vec.shape[-1]:
        warn("Not all entries of `vec` have been used.", stacklevel=1)

    return params_list
