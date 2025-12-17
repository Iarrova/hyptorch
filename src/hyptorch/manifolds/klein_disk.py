import torch

from hyptorch.tensor import squared_norm


class KleinDisk:
    """
    Minimal Klein disk model for hyperbolic averaging operations.

    The Klein disk (also known as the Beltrami-Klein model) represents
    n-dimensional hyperbolic geometry as the interior of a unit ball where
    geodesics are straight Euclidean lines.

    Parameters
    ----------
    curvature : torch.Tensor or float, optional
        The curvature parameter. Default is 1.0.

    Attributes
    ----------
    curvature : torch.Tensor
        The curvature parameter as a tensor.

    Notes
    -----
    This is a minimal implementation focused on the "Klein trick" for
    computing hyperbolic means. For a full hyperbolic manifold with
    exponential/logarithmic maps and Möbius operations, use PoincareBall.

    The Klein model is related to the Poincaré ball by a bijective mapping,
    allowing computations to be performed in whichever model is more convenient.

    Examples
    --------
    >>> # Standalone usage with fixed curvature
    >>> klein = KleinDisk(curvature=1.0)

    >>> # Linked usage with PoincareBall (for trainable curvature)
    >>> from hyptorch.manifolds import PoincareBall
    >>> poincare = PoincareBall(curvature=1.0, trainable_curvature=True)
    >>> klein = KleinDisk(curvature=poincare.curvature)
    """

    def __init__(self, curvature: torch.Tensor | float = 1.0) -> None:
        if isinstance(curvature, torch.Tensor):
            self._curvature = curvature
        else:
            self._curvature = torch.as_tensor(curvature, dtype=torch.float32)

    @property
    def curvature(self) -> torch.Tensor:
        """
        Get the curvature parameter.

        Returns
        -------
        torch.Tensor
            The curvature parameter as a tensor.
        """
        return self._curvature

    def lorentz_factor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Lorentz factor for points in the Klein disk.

        The Lorentz factor is the scaling factor that relates the Klein metric
        to the hyperbolic metric, analogous to the conformal factor in the
        Poincaré model.

        Parameters
        ----------
        x : torch.Tensor
            Points in the Klein disk. Shape (..., dim).

        Returns
        -------
        torch.Tensor
            Lorentz factor at each point.

        Notes
        -----
        The Lorentz factor for a point :math:`\\mathbf{x}` in the Klein model
        with curvature :math:`c` is:

        .. math::
            \\gamma_{\\mathbf{x}}^c = \\frac{1}{\\sqrt{1 - c \\|\\mathbf{x}\\|^2}}

        This factor approaches infinity as points approach the boundary of
        the Klein disk, reflecting the infinite distance to the boundary
        in hyperbolic geometry.

        Examples
        --------
        >>> klein = KleinDisk(curvature=1.0)
        >>> x = torch.tensor([[0.3, 0.4], [0.0, 0.0]])
        >>> gamma = klein.lorentz_factor(x)
        """
        return 1 / torch.sqrt(1 - self.curvature * squared_norm(x))

    def mean(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hyperbolic mean (Einstein midpoint) of points in the Klein disk.

        This method computes the weighted average of points in the Klein model,
        where weights are determined by the Lorentz factors. This is the
        hyperbolic generalization of the Euclidean centroid.

        Parameters
        ----------
        x : torch.Tensor
            Points in the Klein disk.

        Returns
        -------
        torch.Tensor
            Hyperbolic mean in the Klein disk.

        Notes
        -----
        The hyperbolic mean in the Klein model (Einstein midpoint) is:

        .. math::

            \\bar{\\mathbf{x}} = \\frac{\\sum_i \\gamma_i \\mathbf{x}_i}{\\sum_i \\gamma_i}

        where :math:`\\gamma_i` is the Lorentz factor at point :math:`\\mathbf{x}_i`.

        This formula arises from the fact that the Klein model is the projective
        model of hyperbolic space, and averaging must account for the non-uniform
        metric via the Lorentz factors.

        Examples
        --------
        >>> klein = KleinDisk(curvature=1.0)
        >>> # Batch of 10 points in 5D
        >>> points = torch.randn(10, 5) * 0.3
        >>> mean = klein.mean(points, dim=0)  # Shape: (5,)

        >>> # Multiple batches
        >>> points = torch.randn(32, 10, 5) * 0.3  # 32 batches of 10 points
        >>> means = klein.mean(points, dim=1)  # Shape: (32, 5)
        """
        gamma = self.lorentz_factor(x)
        weighted_sum = torch.sum(gamma * x, dim=0, keepdim=True)
        gamma_sum = torch.sum(gamma, dim=0, keepdim=True)
        return weighted_sum / gamma_sum
