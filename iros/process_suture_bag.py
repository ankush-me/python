#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("task")
parser.add_argument("part_index",type=int)
parser.add_argument("--segment_index", type=int, default=0)
args = parser.parse_args()

import lfd
import iros
from lfd import bag_proc as bp
import rosbag
import rospy
import os.path as osp
from jds_utils.yes_or_no import yes_or_no
import brett2.ros_utils as ru
from jds_utils.colorize import colorize
import os, sys
import yaml
import h5py
import numpy as np
import cv2
import simple_clicker as sc

# find data files, files to save to
IROS_DATA_DIR = os.getenv("IROS_DATA_DIR")
IROS_DIR = osp.dirname(iros.__file__)

task_file = osp.join(IROS_DIR, "suture_demos2.yaml")

with open(osp.join(IROS_DIR, task_file),"r") as fh:
    task_info = yaml.load(fh)

bag = task_info[args.task][args.part_index]["demo_bag"]
#bag = "/mnt/storage/mtayfred/data/bags2/2loopknotseg3.bag"
bag = rosbag.Bag(bag)

jtf = osp.join(IROS_DATA_DIR, args.task, 'joint_trajectories', 'pt' + str(args.part_index))
kpf = osp.join(IROS_DATA_DIR, args.task, 'keypoints', 'pt' + str(args.part_index))
pcf = osp.join(IROS_DATA_DIR, args.task, 'point_clouds', 'pt' + str(args.part_index))

#jtf = osp.join(IROS_DATA_DIR, args.task, 'temp_joint_trajectories', 'pt' + str(args.part_index))
#kpf = osp.join(IROS_DATA_DIR, args.task, 'temp_keypoints', 'pt' + str(args.part_index))
#pcf = osp.join(IROS_DATA_DIR, args.task, 'temp_point_clouds', 'pt' + str(args.part_index))

### extract kinematics info from bag
def extract_kinematics(np_file, info):
    
    for i in range(len(info)):
        savefile = np_file + '/seg%s'%i
        print 'Writing to', np_file

        if osp.exists(savefile + '_larm.npy'):
            if yes_or_no('/seg%s already exists. Overwrite?'%i):
                os.remove(savefile + '_larm.npy')
                os.remove(savefile + '_rarm.npy')
                os.remove(savefile + '_lgrip.npy')
                os.remove(savefile + '_rgrip.npy')
            else:
                print 'Aborting...'
                sys.exit(1)

        larm = info[i]["leftarm"]
        rarm = info[i]["rightarm"]
        lgrip = info[i]["l_gripper_joint"]
        rgrip = info[i]["r_gripper_joint"]

        np.save(savefile + '_larm.npy' , np.array(larm))
        np.save(savefile + '_rarm.npy' , np.array(rarm))
        np.save(savefile + '_lgrip.npy' , np.array(lgrip))
        np.save(savefile + '_rgrip.npy' , np.array(rgrip))

    print 'Saved all trajectories for %i segments'%i
    
       
# creates the text for the yaml file for the demo
def get_new_demo_entry_text(segment, hole1, hole2, tcut, mcut, bcut, needle):

    new_entry_text = """        
        segment%(seg_num)s:
            hole1 center:  %(hole1)s
            hole2 center:  %(hole2)s
            top of cut:    %(tcut)s
            middle of cut: %(mcut)s
            bottom of cut: %(bcut)s
            needle tip:    %(needle)s
""" % dict(
            seg_num = segment,
            hole1 = hole1,
            hole2 = hole2,
            tcut = tcut,
            mcut = mcut,
            bcut = bcut,
            needle = needle,
            )

    return new_entry_text


# appends the entry text to the yaml file
def add_new_entry_to_yaml_file(data_dir, new_entry_text):
    demofile = open(data_dir + "/suture_demos.yaml", "a")
    demofile.write(new_entry_text)
    demofile.flush()
    os.fsync(demofile.fileno())
    demofile.close()
    
    
def get_last_kp_loc(all_past_keypts, desired_keypt, current_seg):        
    
    search_seg = current_seg - 1
       
    while(True):
        if search_seg < 0:
            print "Reached beginning of execution and couldn't find desired keypoint! Aborting..."
            sys.exit(1)            
        else:
            search_seg_names = all_past_keypts[search_seg]["names"]
            search_seg_locs = all_past_keypts[search_seg]["locations"]
        
            for k in range(len(search_seg_names)):
                if search_seg_names[k] == desired_keypt:
                    kp_loc = search_seg_locs[k]
                    kp_found = True
            
        if kp_found:        
            return kp_loc, search_seg
        else:
            search_seg -= 1
            
            
#######################################

print 'getting bag info...'

button_presses = bp.get_button_presses(bag)
all_times = bp.read_button_presses(button_presses)
start_times = [t for t, action in all_times if action == 'start']
stop_times = [t for t, action in all_times if action == 'stop']
look_times = [t for t, action in all_times if action == 'look']

assert len(start_times) == len(stop_times)
SEGNUM = len(look_times)

if yes_or_no(colorize('Do you want to extract kinematics info from bag file?', 'blue')):
    link_names = ["r_gripper_tool_frame", "l_gripper_tool_frame"]             
    kinematics_info = bp.extract_kinematics_from_bag(bag, link_names)    
    times = kinematics_info["time"]

    start_inds = np.searchsorted(times, start_times)
    stop_inds = np.searchsorted(times, stop_times)

    forced_segs_info = []
    for start_ind, stop_ind in zip(start_inds, stop_inds):
        seg = bp.extract_segment(kinematics_info, start_ind, stop_ind)
        forced_segs_info.append(seg)
    
    extract_kinematics(jtf, forced_segs_info)         

    raw_input("Extracted segments. Press enter to continue")   

window_name = "Find Keypoints"
keypt_list = ['lh', 'rh', 'tc', 'mc', 'bc', 'ne', 'nt', 'ntt', 'stand', 'rzr']
demo_keypts = {}   

sstart = args.segment_index

# determine which keypoints matter for each segment based on user input
print 'Extracting look_time point clouds...'
look_clouds = bp.get_transformed_clouds(bag, look_times)

for s in range(sstart, SEGNUM):
    print colorize('look time for segment %s'%s, 'blue', bold=True, highlight=True)

    demo_keypts[s] = {}
    demo_keypts[s]["names"] = []
    demo_keypts[s]["locations"] = []    
    
    k = 0

    while (True):
        sc.show_pointclouds(look_clouds[s], window_name)
        kp = raw_input("Which key points are important for this segment? choices are: %s. (please only enter one key point at a time): "%str(keypt_list))
        
        if kp not in keypt_list:
            print 'Incorrect input. Try again!'
            continue  
                 
        elif kp in ['lh', 'rh', 'bc', 'mc', 'tc']:
            if kp == 'lh': demo_keypts[s]["names"].append('left_hole')
            elif kp == 'rh': demo_keypts[s]["names"].append('right_hole')
            elif kp == 'bc': demo_keypts[s]["names"].append('bot_cut')
            elif kp == 'mc': demo_keypts[s]["names"].append('mid_cut')
            elif kp == 'tc': demo_keypts[s]["names"].append('top_cut')            

            print colorize("Looking for %s..."%demo_keypts[s]["names"][k], 'green', bold=True)
            
            xyz_tf = look_clouds[s][0].copy()
            rgb_plot = look_clouds[s][1].copy()
            
            if k < 1:
                np.save(osp.join(pcf, 'seg%s_xyz_tf.npy'%s), xyz_tf)
                np.save(osp.join(pcf, 'seg%s_rgb_pl.npy'%s), rgb_plot)            

            kp_loc = sc.find_kp(demo_keypts[s]["names"][k], xyz_tf, rgb_plot, window_name)
            
            if (np.isnan(np.asarray(kp_loc))).all():
                last_loc, found_seg = get_last_kp_loc(demo_keypts, demo_keypts[s]["names"][k], s)
                print "occluded key point %s found in segment %s at location %s"%(demo_keypts[s]["names"][k], found_seg, last_loc)
                kp_loc = last_loc            
                                                          

        elif kp in ['stand', 'rzr', 'ne']:
            xyz_tfs = []
            rgb_plots = []          
            num_clouds = 10
            
            if kp == 'ne': demo_keypts[s]["names"].append('needle_end') 
            elif kp == 'rzr': demo_keypts[s]["names"].append('razor') 
            elif kp == 'stand': demo_keypts[s]["names"].append('needle_stand')            
            
            while True:
                kp_look_times = []  
                
                print colorize("Looking for %s using High Red Block..."%demo_keypts[s]["names"][k], 'green', bold=True)          
                print 'getting %s point clouds...'%num_clouds                

                for t in range(num_clouds):
                    kp_look_times.append(look_times[s] + t)
                    
                kp_clouds = bp.get_transformed_clouds(bag, kp_look_times)
    
                for i in range(num_clouds):
                    xyz_tfs.append(kp_clouds[i][0].copy())
                    rgb_plots.append(kp_clouds[i][1].copy())
                
                if k < 1:    
                    np.save(osp.join(pcf, 'seg%s_xyz_tfs.npy'%s), xyz_tfs)
                    np.save(osp.join(pcf, 'seg%s_rgb_pls.npy'%s), rgb_plots)                
                    
                kp_loc, valid_pts = sc.find_red_block(xyz_tfs, rgb_plots, window_name)
                del kp_look_times
                if (valid_pts > 0) or (num_clouds >= 20):                                       
                    break
                else: 
                    print colorize("Couldn't find the keypoint! Trying again with more point clouds", "red", True, True)
                    num_clouds += 5
            
            del xyz_tfs
            del rgb_plots            
            
        elif kp == 'nt':
            xyz_tfs = []
            rgb_plots = []            
            num_clouds = 10
            
            demo_keypts[s]["names"].append('needle_tip')

            while(True):
                needle_look_times = []
                
                print colorize("Looking for Needle Tip...", 'green', bold=True)          
                print 'getting %s needle point clouds...'%num_clouds             
                
                for t in range(num_clouds):
                    needle_look_times.append(look_times[s] + t)
                              
                needle_clouds = bp.get_transformed_clouds(bag, needle_look_times)
    
                for i in range(num_clouds):
                    xyz_tfs.append(needle_clouds[i][0].copy())
                    rgb_plots.append(needle_clouds[i][1].copy())
                
                if k < 1:
                    np.save(osp.join(pcf, 'seg%s_xyz_tfs.npy'%s), xyz_tfs)
                    np.save(osp.join(pcf, 'seg%s_rgb_pls.npy'%s), rgb_plots)                 
                
                kp_loc = sc.find_needle_tip(xyz_tfs, rgb_plots, window_name, 'process')
                if yes_or_no("Is this a good point?"):
                    break
                else:
                    del needle_look_times
                    num_clouds += 5
                    print ("Trying again with more point clouds.")                                                                        
            
            del xyz_tfs
            del rgb_plots  

        elif kp == 'ntt':
            xyz_tfs = []
            rgb_plots = []
            needle_look_times = []            
            num_clouds = 10
            
            demo_keypts[s]["names"].append('tip_transform')
            kp_loc = ((0,0,0))
            
            print colorize("Looking for Needle Tip...", 'green', bold=True)          

            while True:
                print 'getting %s needle point clouds...'%num_clouds              
                
                for t in range(num_clouds):
                    needle_look_times.append(look_times[s] + t)
                             
                needle_clouds = bp.get_transformed_clouds(bag, needle_look_times)
    
                for i in range(num_clouds):
                    xyz_tfs.append(needle_clouds[i][0].copy())
                    rgb_plots.append(needle_clouds[i][1].copy())
                    
                if k < 1:                
                    np.save(osp.join(pcf, 'seg%s_xyz_tfs.npy'%s), xyz_tfs)
                    np.save(osp.join(pcf, 'seg%s_rgb_pls.npy'%s), rgb_plots)                 
                    
                needle_loc = sc.find_needle_tip(xyz_tfs, rgb_plots, window_name, 'process')
                
                if yes_or_no("Is this a good point?"):
                    break
                else:
                    del needle_look_times
                    num_clouds += 5
                    print ("Trying again with more point clouds.")            
                                              
            np.save(osp.join(kpf, 'seg%s_needle_world_loc.npy'%s), needle_loc)                                     
            
            del xyz_tfs
            del rgb_plots 
            
        demo_keypts[s]["locations"].append(kp_loc)
            
        if yes_or_no("Is there another key point for this segment?"):
            k += 1
            continue    
        else:
            np.save(osp.join(kpf, 'seg%s_keypoints.npy'%s), demo_keypts[s]["locations"])
            np.save(osp.join(kpf, 'seg%s_keypoints_names.npy'%s), demo_keypts[s]["names"])
            print "Keypoint(s): %s saved to file."%demo_keypts[s]["names"]
            break
            
  

#raw_input("press enter to add new entry to yaml file")

#entry = get_new_demo_entry_text(i, hole1_tracker[i], hole2_tracker[i], tcut_tracker[i], mcut_tracker[i], bcut_tracker[i], needle_tracker[i])
#add_new_entry_to_yaml_file(IROS_DATA_DIR, str(SEGNUM))

raw_input("press enter to finish")
