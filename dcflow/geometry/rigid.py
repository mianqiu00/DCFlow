from dcflow.vendor.openfold.utils.rigid_utils import Rigid, Rotation


def create_rigid(rots, trans):
    return Rigid(rots=Rotation(rot_mats=rots), trans=trans)
