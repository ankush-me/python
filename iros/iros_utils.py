from jds_utils.colorize import colorize
import numpy as np
import jds_utils.math_utils as mu
import trajoptpy
import openravepy
import json

import suturing_logger


def transform_hmats(f, hmats):
    hmats = np.array(hmats)
    oldpts_md = hmats[:,:3,3]
    oldrots_mad = hmats[:,:3,:3]
    newpts_mg, newrots_mgd = f.transform_frames(oldpts_md, oldrots_mad)    
    tf_hmats = hmats.copy()
    tf_hmats[:,:3,:3] = newrots_mgd
    tf_hmats[:,:3,3] = newpts_mg
    return tf_hmats

def translation_matrix(xyz):
    out = np.eye(4)
    out[:3,3] = xyz

def adaptive_resample(x, tol, max_change=-1, min_steps=3):
    """
    resample original signal with a small number of waypoints so that the the sparsely sampled function, 
    when linearly interpolated, deviates from the original function by less than tol at every time

    input:
    x: 2D array in R^(t x k)  where t is the number of timesteps
    tol: tolerance. either a single scalar or a vector of length k
    max_change: max change in the sparsely sampled signal at each timestep
    min_steps: minimum number of timesteps in the new trajectory. (usually irrelevant)

    output:
    new_times, new_x

    assuming that the old signal has times 0,1,2,...,len(x)-1
    this gives the new times, and the new signal
    """

    x = np.asarray(x)
    assert x.ndim == 2

    if np.isscalar(tol): 
        tol = np.ones(x.shape[1])*tol
    else:
        tol = np.asarray(tol)
        assert tol.ndim == 1 and tol.shape[0] == x.shape[1]

    times = np.arange(x.shape[0])

    if max_change == -1: 
        max_change = np.ones(x.shape[1]) * np.inf
    elif np.isscalar(max_change): 
        max_change = np.ones(x.shape[1]) * max_change
    else:
        max_change = np.asarray(max_change)
        assert max_change.ndim == 1 and max_change.shape[0] == x.shape[1]

    dl = mu.norms(x[1:] - x[:-1],1)
    l = np.cumsum(np.r_[0,dl])

    def bad_inds(x1, t1):
        ibad = np.flatnonzero( (np.abs(mu.interp2d(l, l1, x1) - x) > tol).any(axis=1) )
        jbad1 = np.flatnonzero((np.abs(x1[1:] - x1[:-1]) > max_change[None,:]).any(axis=1))
        if len(ibad) == 0 and len(jbad1) == 0: return []
        else:
            lbad = l[ibad]
            jbad = np.unique(np.searchsorted(l1, lbad)) - 1
            jbad = np.union1d(jbad, jbad1)
            return jbad

    l1 = np.linspace(0,l[-1],min_steps)
    for _ in xrange(20):
        x1 = mu.interp2d(l1, l, x)
        bi = bad_inds(x1, l1)
        if len(bi) == 0:
            return np.interp(l1, l, times), x1
        else:
            l1 = np.union1d(l1, (l1[bi] + l1[bi+1]) / 2 )


    raise Exception("couldn't subdivide enough. something funny is going on. check your input data")


def plan_follow_traj(robot, manip_name, link_name, new_hmats, old_traj, other_manip_name = None, other_manip_traj = None):

    #import trajoptpy
    #trajoptpy.SetInteractive(True)
    #import IPython
    #IPython.embed()

    n_steps = len(new_hmats)
    assert old_traj.shape[0] == n_steps
    assert old_traj.shape[1] == 6 # for ravens each arm has 6 dofs (+ 2 for fingers)

    init_traj = old_traj.copy()
    init_traj[0] = robot.GetDOFValues(robot.GetManipulator(manip_name).GetArmIndices())

    request = {
        "basic_info" : {
            "n_steps" : n_steps,
            "manip" : manip_name,
            "start_fixed" : True
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [1]}
        },
        # {
        #     "type" : "collision",
        #     "params" : {"coeffs" : [10],"dist_pen" : [0.025]}
        # }                
        ],
        "constraints" : [
        ],
        "init_info" : {
            #"type":"given_traj",
            "type":"stationary",
            "data":[x.tolist() for x in init_traj]
        }
    }
    if other_manip_name is not None:
        print "omn if"
        request["scene_states"] = []
        other_dof_inds = robot.GetManipulator(other_manip_name).GetArmIndices()

    poses = [openravepy.poseFromMatrix(hmat) for hmat in new_hmats]
    for (i_step,pose) in enumerate(poses):
        request["costs"].append(
            {"type":"pose",
             "params":{
                "xyz":pose[4:7].tolist(),
                "wxyz":pose[0:4].tolist(),
                "link":link_name,
                "timestep":i_step,
                "pos_coeffs":[100,100,100],
                "rot_coeffs":[10,10,10]
             }
            })
        if other_manip_name is not None:
            request["scene_states"].append(
                {"timestep": i_step, "obj_states": [{"name": "ravens", "dof_vals":other_manip_traj[i_step], "dof_inds":other_dof_inds}] })

    env = robot.GetEnv()
    s = json.dumps(request)
    prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
    result = trajoptpy.OptimizeProblem(prob)  # do optimization
    traj = result.GetTraj()    
    return traj


def plan_follow_traj2(robot, manip_name, link1_name, new_hmats1, link2_name, new_hmats2, old_traj, segment_info=None, other_manip_name = None, other_manip_traj = None):

    #for i in xrange(len(new_hmats1)):
    #    new_hmats1[i,:3,:3] = new_hmats1[i,:3,:3].T
    #    new_hmats2[i,:3,:3] = new_hmats2[i,:3,:3].T
    #new_hmats1[:,:3,:3] = new_hmats1[:,:3:-1,3:-1:-1]
    #new_hmats2[:,:3,:3] = new_hmats2[:,:3:-1,3:-1:-1]

    #import trajoptpy
    #trajoptpy.SetInteractive(True)
    #import IPython
    #IPython.embed()

    n_steps = len(new_hmats1)
    assert old_traj.shape[0] == n_steps
    assert len(new_hmats1)   == len(new_hmats2)
    assert old_traj.shape[1] == 6 # for ravens each arm has 6 dofs (+ 2 for fingers)

    init_traj = old_traj.copy()
    init_traj[0] = robot.GetDOFValues(robot.GetManipulator(manip_name).GetArmIndices())

    request = {
        "basic_info" : {
            "n_steps"     : n_steps,
            "manip"       : manip_name,
            "start_fixed" : True
        },
        "costs" : [
        {
            "type" : "joint_vel",
            "params": {"coeffs" : [1]}
        },
        {
            "type" : "collision",
            "params" : {"continuous": True, "coeffs" : [100],"dist_pen" : [0.001]}
        }                
        ],
        "constraints" : [
        ],
        "init_info" : {
            "type":"given_traj",
            #"type":"stationary",
            "data":[x.tolist() for x in init_traj]
        }
    }
    if other_manip_name is not None:
        request["scene_states"] = []
        other_dof_inds = robot.GetManipulator(other_manip_name).GetArmIndices()

    r = 2.8
    pos_coeffs = [100,100,100]
    rot_coeffs = [r,r,r]

    #trajoptpy.SetInteractive(True)

    poses1 = [openravepy.poseFromMatrix(hmat) for hmat in new_hmats1]
    for (i_step,pose) in enumerate(poses1):
        request["costs"].append(
            {"type":"pose",
             "params":{
                "xyz":pose[4:7].tolist(),
                "wxyz":pose[0:4].tolist(),
                "link":link1_name,
                "timestep":i_step,
                "pos_coeffs":pos_coeffs,
                "rot_coeffs":rot_coeffs
             }
            })
        if other_manip_name is not None:
            request["scene_states"].append(
                {"timestep": i_step, "obj_states": [{"name": "ravens", "dof_vals":other_manip_traj[i_step], "dof_inds":other_dof_inds}] })


    poses2 = [openravepy.poseFromMatrix(hmat) for hmat in new_hmats2]
    for (i_step,pose) in enumerate(poses2):
        request["costs"].append(
            {"type":"pose",
             "params":{
                "xyz":pose[4:7].tolist(),
                "wxyz":pose[0:4].tolist(),
                "link":link2_name,
                "timestep":i_step,
                "pos_coeffs":pos_coeffs,
                "rot_coeffs":rot_coeffs
             }
            })

    env = robot.GetEnv()
    s = json.dumps(request)
    prob = trajoptpy.ConstructProblem(s, env) # create object that stores optimization problem
    trajoptpy.GetCollisionChecker(robot.GetEnv()).SetContactDistance(.001)
    result = trajoptpy.OptimizeProblem(prob)  # do optimization
    traj = result.GetTraj()


    # calculate the position errors at each time-step in the segment:
    saver = openravepy.RobotStateSaver(robot)
    arm_inds = robot.GetManipulator(manip_name).GetArmIndices()
    link1 = robot.GetLink(link1_name)
    link2 = robot.GetLink(link2_name)
    pos_errs1 = []
    pos_errs2 = []
    for i_step in xrange(1,n_steps):
        row = traj[i_step]
        robot.SetDOFValues(row, arm_inds)
        tf1 = openravepy.matrixFromPose(link1.GetTransform())
        tf2 = openravepy.matrixFromPose(link2.GetTransform())

        pos_err1 = np.linalg.inv(new_hmats1[i_step]).dot(tf1)
        pos_err2 = np.linalg.inv(new_hmats2[i_step]).dot(tf2)

        pos_errs1.append(pos_err1)
        pos_errs2.append(pos_err2)

    pos_errs = np.array([pos_errs1, pos_errs2])
    traj_res = [result.GetCosts(), result.GetConstraints()]
    
    suturing_logger.log(segment_info, pos_errs, traj_res)

    return traj
