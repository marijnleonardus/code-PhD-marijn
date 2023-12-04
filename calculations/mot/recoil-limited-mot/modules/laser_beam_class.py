# define laser beams using pyLCP class
# use the parent class laserBeams

from pylcp import laserBeams
from pylcp.fields import infinitePlaneWaveBeam
from scipy.spatial.transform import Rotation
import numpy as np


class AngledMOTBeams(laserBeams):
    """
    A collection of laser beams for 6-beam MOT

    This class uses 4 angled MOT beams in the X,Z plane 
    and horitonal MOT beams in the y direction

    The standard geometry is to generate counter-progagating beams along all
    orthogonal axes :math:`(\\hat{x}, \\hat{y}, \\hat{z})`.

    Parameters
    ----------
    k : float, optional
        Magnitude of the k-vector for the six laser beams.  Default: 1
    pol : int or float, optional
        Sign of the circular polarization for the beams moving along
        :math:`\\hat{z}`.  Default: +1.  Orthogonal beams have opposite
        polarization by default.
    rotation_angles : array_like
        List of angles to define a rotated MOT.  Default: [0., 0., 0.]
    rotation_spec : str
        String to define the convention of the Euler rotations.  Default: 'ZYZ'
    beam_type : pylcp.laserBeam or subclass
        Type of beam to generate.
    **kwargs :
        other keyword arguments to pass to beam_type
    """
    def __init__(self, k=1, pol=+1, rotation_angles=[0., 0., 0.],
                 rotation_spec='ZYZ', beam_type=infinitePlaneWaveBeam, **kwargs):
        super().__init__()

        rot_mat = Rotation.from_euler(rotation_spec, rotation_angles).as_matrix()

        kvecs = [np.array([.5*np.sqrt(3), 0., -.5]), np.array([-0.5*np.sqrt(3), 0., +.5]),
                 np.array([0,  1.,  0.]), np.array([ 0., -1.,  0.]),
                 np.array([.5*np.sqrt(3), 0., +.5]), np.array([-0.5*np.sqrt(3), 0., -.5])]
        pols = [-pol, -pol, +pol, +pol, -pol, -pol,]

        for kvec, pol in zip(kvecs, pols):
            self.add_laser(beam_type(kvec=rot_mat @ (k*kvec), pol=pol, **kwargs))
