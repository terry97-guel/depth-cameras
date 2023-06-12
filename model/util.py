import math,time
import numpy as np
import shapely as sp # handle polygon
from shapely import Polygon,LineString,Point # handle polygons
from scipy.spatial.distance import cdist

def rot_mtx(deg):
    """
        2 x 2 rotation matrix
    """
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R

def pr2t(p,R):
    """ 
        Convert pose to transformation matrix 
    """
    p0 = p.ravel() # flatten
    T = np.block([
        [R, p0[:, np.newaxis]],
        [np.zeros(3), 1]
    ])
    return T

def TransInv(T):
    """
        Inverts a homogeneous transformation matrix
    """
    p, R = t2pr(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]

def t2pr(T):
    """
        T to p and R
    """   
    p = T[:3,3]
    R = T[:3,:3]
    return p,R

def t2p(T):
    """
        T to p 
    """   
    p = T[:3,3]
    return p

def t2r(T):
    """
        T to R
    """   
    R = T[:3,:3]
    return R    

def so3ToVec(so3mat):
    """
        Converts an so(3) representation to a 3-vector
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def VecToso3(omg):
    """
        Converts a 3-vector to an so(3) representation
    """
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])
def NearZero(z):
    """
        Determines whether a scalar is small enough to be treated as zero
    """
    return abs(z) < 1e-6

def AxisAng3(expc3):
    """
        Converts a 3-vector of exponential coordinates for rotation into axis-angle form
    """
    return ((expc3 / np.linalg.norm(expc3)), np.linalg.norm(expc3))

def MatrixLog3(R):
    """
        Computes the matrix logarithm of a rotation matrix
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def MatrixExp3(so3mat):
    """
        Computes the matrix exponential of a matrix in so(3)
    """
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def MatrixExp6(se3mat):
    """
        Computes the matrix exponential of an se3 representation of
    """
    se3mat = np.array(se3mat)
    omgtheta = so3ToVec(se3mat[0: 3, 0: 3])
    if NearZero(np.linalg.norm(omgtheta)):
        return np.r_[np.c_[np.eye(3), se3mat[0: 3, 3]], [[0, 0, 0, 1]]]
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = se3mat[0: 3, 0: 3] / theta
        return np.r_[np.c_[MatrixExp3(se3mat[0: 3, 0: 3]),
                           np.dot(np.eye(3) * theta \
                                  + (1 - np.cos(theta)) * omgmat \
                                  + (theta - np.sin(theta)) \
                                    * np.dot(omgmat,omgmat),
                                  se3mat[0: 3, 3]) / theta],
                     [[0, 0, 0, 1]]]

def MatrixLog6(T):
    """
        Computes the matrix logarithm of a homogeneous transformation matrix
    """
    p, R = t2p(T)
    omgmat = MatrixLog3(R)
    if np.array_equal(omgmat, np.zeros((3, 3))):
        return np.r_[np.c_[np.zeros((3, 3)),
                           [T[0][3], T[1][3], T[2][3]]],
                     [[0, 0, 0, 0]]]
    else:
        theta = np.arccos((np.trace(R) - 1) / 2.0)
        return np.r_[np.c_[omgmat,
                           np.dot(np.eye(3) - omgmat / 2.0 \
                           + (1.0 / theta - 1.0 / np.tan(theta / 2.0) / 2) \
                              * np.dot(omgmat,omgmat) / theta,[T[0][3],
                                                               T[1][3],
                                                               T[2][3]])],
                     [[0, 0, 0, 0]]]


def rpy2r(rpy):
    """
        roll,pitch,yaw in radian to R
    """
    roll  = rpy[0]
    pitch = rpy[1]
    yaw   = rpy[2]
    Cphi  = np.math.cos(roll)
    Sphi  = np.math.sin(roll)
    Cthe  = np.math.cos(pitch)
    Sthe  = np.math.sin(pitch)
    Cpsi  = np.math.cos(yaw)
    Spsi  = np.math.sin(yaw)
    R     = np.array([
        [Cpsi * Cthe, -Spsi * Cphi + Cpsi * Sthe * Sphi, Spsi * Sphi + Cpsi * Sthe * Cphi],
        [Spsi * Cthe, Cpsi * Cphi + Spsi * Sthe * Sphi, -Cpsi * Sphi + Spsi * Sthe * Cphi],
        [-Sthe, Cthe * Sphi, Cthe * Cphi]
    ])
    assert R.shape == (3, 3)
    return R

def r2rpy(R,unit='rad'):
    """
        Rotation matrix to roll,pitch,yaw in radian
    """
    roll  = math.atan2(R[2, 1], R[2, 2])
    pitch = math.atan2(-R[2, 0], (math.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    yaw   = math.atan2(R[1, 0], R[0, 0])
    if unit == 'rad':
        out = np.array([roll, pitch, yaw])
    elif unit == 'deg':
        out = np.array([roll, pitch, yaw])*180/np.pi
    else:
        out = None
        raise Exception("[r2rpy] Unknown unit:[%s]"%(unit))
    return out    

def r2w(R):
    """
        R to \omega
    """
    el = np.array([
            [R[2,1] - R[1,2]],
            [R[0,2] - R[2,0]], 
            [R[1,0] - R[0,1]]
        ])
    norm_el = np.linalg.norm(el)
    if norm_el > 1e-10:
        w = np.arctan2(norm_el, np.trace(R)-1) / norm_el * el
    elif R[0,0] > 0 and R[1,1] > 0 and R[2,2] > 0:
        w = np.array([[0, 0, 0]]).T
    else:
        w = np.math.pi/2 * np.array([[R[0,0]+1], [R[1,1]+1], [R[2,2]+1]])
    return w.flatten()

def r2quat(R):
    """ 
        Convert Rotation Matrix to Quaternion.  See rotation.py for notes 
        (https://gist.github.com/machinaut/dab261b78ac19641e91c6490fb9faa96)
    """
    R = np.asarray(R, dtype=np.float64)
    Qxx, Qyx, Qzx = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    Qxy, Qyy, Qzy = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    Qxz, Qyz, Qzz = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(R.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q

def trim_scale(x,th):
    """
        Trim scale
    """
    x         = np.copy(x)
    x_abs_max = np.abs(x).max()
    if x_abs_max > th:
        x = x*th/x_abs_max
    return x

def soft_squash(x,x_min=-1,x_max=+1,margin=0.1):
    """
        Soft squashing numpy array
    """
    def th(z,m=0.0):
        # thresholding function 
        return (m)*(np.exp(2/m*z)-1)/(np.exp(2/m*z)+1)
    x_in = np.copy(x)
    idxs_upper = np.where(x_in>(x_max-margin))
    x_in[idxs_upper] = th(x_in[idxs_upper]-(x_max-margin),m=margin) + (x_max-margin)
    idxs_lower = np.where(x_in<(x_min+margin))
    x_in[idxs_lower] = th(x_in[idxs_lower]-(x_min+margin),m=margin) + (x_min+margin)
    return x_in    

def soft_squash_multidim(
    x      = np.random.randn(100,5),
    x_min  = -np.ones(5),
    x_max  = np.ones(5),
    margin = 0.1):
    """
        Multi-dim version of 'soft_squash' function
    """
    x_squash = np.copy(x)
    dim      = x.shape[1]
    for d_idx in range(dim):
        x_squash[:,d_idx] = soft_squash(
            x=x[:,d_idx],x_min=x_min[d_idx],x_max=x_max[d_idx],margin=margin)
    return x_squash 

def kernel_se(X1,X2,hyp={'g':1,'l':1}):
    """
        Squared exponential (SE) kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    return K

def kernel_levse(X1,X2,L1,L2,hyp={'g':1,'l':1}):
    """
        Leveraged SE kernel function
    """
    K = hyp['g']*np.exp(-cdist(X1,X2,'sqeuclidean')/(2*hyp['l']*hyp['l']))
    L = np.cos(np.pi/2.0*cdist(L1,L2,'cityblock'))
    return np.multiply(K,L)

def is_point_in_polygon(point,polygon):
    """
        Is the point inside the polygon
    """
    if isinstance(point,np.ndarray):
        point_check = Point(point)
    else:
        point_check = point
    return sp.contains(polygon,point_check)

def is_point_feasible(point,obs_list):
    """
        Is the point feasible w.r.t. obstacle list
    """
    result = is_point_in_polygon(point,obs_list) # is the point inside each obstacle?
    if sum(result) == 0:
        return True
    else:
        return False

def is_point_to_point_connectable(point1,point2,obs_list):
    """
        Is the line connecting two points connectable
    """
    result = sp.intersects(LineString([point1,point2]),obs_list)
    if sum(result) == 0:
        return True
    else:
        return False
    
class TicTocClass(object):
    """
        Tic toc
    """
    def __init__(self,name='tictoc',print_every=1):
        """
            Initialize
        """
        self.name        = name
        self.time_start  = time.time()
        self.time_end    = time.time()
        self.print_every = print_every

    def tic(self):
        """
            Tic
        """
        self.time_start = time.time()

    def toc(self,str=None,cnt=0,VERBOSE=True):
        """
            Toc
        """
        self.time_end = time.time()
        self.time_elapsed = self.time_end - self.time_start
        if VERBOSE:
            if self.time_elapsed <1.0:
                time_show = self.time_elapsed*1000.0
                time_unit = 'ms'
            elif self.time_elapsed <60.0:
                time_show = self.time_elapsed
                time_unit = 's'
            else:
                time_show = self.time_elapsed/60.0
                time_unit = 'min'
            if (cnt % self.print_every) == 0:
                if str is None:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (self.name,time_show,time_unit))
                else:
                    print ("%s Elapsed time:[%.2f]%s"%
                        (str,time_show,time_unit))

def get_interp_const_vel_traj(traj_anchor,vel=1.0,HZ=100,ord=np.inf):
    """
        Get linearly interpolated constant velocity trajectory
    """
    L = traj_anchor.shape[0]
    D = traj_anchor.shape[1]
    dists = np.zeros(L)
    for tick in range(L):
        if tick > 0:
            p_prev,p_curr = traj_anchor[tick-1,:],traj_anchor[tick,:]
            dists[tick] = np.linalg.norm(p_prev-p_curr,ord=ord)
    times_anchor = np.cumsum(dists/vel) # [L]
    L_interp = int(times_anchor[-1]*HZ)
    times_interp = np.linspace(0,times_anchor[-1],L_interp) # [L_interp]
    traj_interp = np.zeros((L_interp,D)) # [L_interp x D]
    for d_idx in range(D):
        traj_interp[:,d_idx] = np.interp(times_interp,times_anchor,traj_anchor[:,d_idx])
    return times_interp,traj_interp

def meters2xyz(depth_img,cam_matrix):
    """
        Scaled depth image to pointcloud
    """
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]
    
    height = depth_img.shape[0]
    width = depth_img.shape[1]
    indices = np.indices((height, width),dtype=np.float32).transpose(1,2,0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([z_e, -x_e, -y_e], axis=-1) # [H x W x 3] 
    return xyz_img # [H x W x 3]

def compute_view_params(camera_pos,target_pos,up_vector=np.array([0,0,1])):
    """Compute azimuth, distance, elevation, and lookat for a viewer given camera pose in 3D space.

    Args:
        camera_pos (np.ndarray): 3D array of camera position.
        target_pos (np.ndarray): 3D array of target position.
        up_vector (np.ndarray): 3D array of up vector.

    Returns:
        tuple: Tuple containing azimuth, distance, elevation, and lookat values.
    """
    # Compute camera-to-target vector and distance
    cam_to_target = target_pos - camera_pos
    distance = np.linalg.norm(cam_to_target)

    # Compute azimuth and elevation
    azimuth = np.arctan2(cam_to_target[1], cam_to_target[0])
    azimuth = np.rad2deg(azimuth) # [deg]
    elevation = np.arcsin(cam_to_target[2] / distance)
    elevation = np.rad2deg(elevation) # [deg]

    # Compute lookat point
    lookat = target_pos

    # Compute camera orientation matrix
    zaxis = cam_to_target / distance
    xaxis = np.cross(up_vector, zaxis)
    yaxis = np.cross(zaxis, xaxis)
    cam_orient = np.array([xaxis, yaxis, zaxis])

    # Return computed values
    return azimuth, distance, elevation, lookat

def sample_xyzs(n_sample,x_range=[0,1],y_range=[0,1],z_range=[0,1],min_dist=0.1):
    """
        Sample a point in three dimensional space with the minimum distance between points
    """
    xyzs = np.zeros((n_sample,3))
    for p_idx in range(n_sample):
        while True:
            x_rand = np.random.uniform(low=x_range[0],high=x_range[1])
            y_rand = np.random.uniform(low=y_range[0],high=y_range[1])
            z_rand = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x_rand,y_rand,z_rand])
            if p_idx == 0: break
            devc = cdist(xyz.reshape((-1,3)),xyzs[:p_idx,:].reshape((-1,3)),'euclidean')
            if devc.min() > min_dist: break # minimum distance between objects
        xyzs[p_idx,:] = xyz
    return xyzs

