import numpy as np

import pylcp


class XquadrupoleMagneticField(magField):
    """
    Spherical quadrupole  magnetic field

    Represents a magnetic field of the form

    .. math::
      \\mathbf{B} = \\alpha\\left(- \\frac{x\\hat{x}}{2} - \\frac{y\\hat{y}}{2} + z\\hat{z}\\right)

    Parameters
    ----------
    alpha : float
        strength of the magnetic field gradient.
    """

    def __init__(self, alpha, eps=1e-5):
        super().__init__(lambda R, t: alpha*np.array([-0.5*R[0], -0.5*R[1], R[2]]))
        self.alpha = alpha

        self.constant_grad_field = alpha*\
            np.array([[-0.5, 0., 0.], [0., -0.5, 0.], [0., 0., 1.]])

    def gradField(self, R=np.array([0., 0., 0.]), t=0):
        """
        Full spaitial derivative of the magnetic field at R and t:

        Parameters
        ----------
        R : array_like, size (3,), optional
            vector of the position at which to return the kvector.  By default,
            the origin.
        t : float, optional
            time at which to return the k-vector.  By default, t=0.

        Returns
        -------
        dB : array_like, shape (3, 3)
            the full gradient of the magnetic field, with elements

            .. math::
              \\begin{pmatrix}
                -\\alpha/2 & 0 & 0 \\\\
                0 & -\\alpha/2 & 0 \\\\
                0 & 0 & \\alpha \\\\
              \\end{pmatrix}
        """
        return self.constant_grad_field
    