# Log segment information:
# ========================
#   - command-line string == perturbation params
#   - scene-recording file name
#   - for each segment:
#        - num time-steps
#        - warping cost
#        - trajopt costs/ constraints
#        - position error for each time-step
#
import cPickle

segments_info = []
num_segs      = None
curr_seg      = 0
scene_recording_fname = None
seg_call_num  = 0
curr_seg_info = {}
save_path     = None
perturbations = None

def log (seg_info, pos_errs, traj_res):
    global segments_info, num_segs, curr_seg, scene_recording_fname
    global seg_call_num, curr_seg_info, save_path, perturbations

    seg_call_num += 1

    #=============== demo-level info ===========:
    if perturbations == None:
        perturbations = seg_info['perturbations']

    if num_segs == None:
       num_segs = seg_info['num_segs']
    
    if scene_recording_fname == None:
        scene_recording_fname = seg_info['recording_fname']

    if save_path==None and scene_recording_fname != None:
        save_path = scene_recording_fname[:-4] + '-costs.txt'
    #===========================================

    # start new segment-level information
    if seg_call_num==2:
        curr_seg += 1
        seg_call_num = 0
        segments_info.append(curr_seg_info)
        curr_seg_info = {}

    #========= segment level information ================
    if not curr_seg_info.has_key('warp_costs'):
        curr_seg_info['warp_costs'] = seg_info['warp_costs']
    
    if not curr_seg_info.has_key('n_steps'):
        curr_seg_info['n_steps'] = pos_errs.shape[1]
        
    if seg_call_num==0:
        curr_seg_info['rarm_costs'] = (pos_errs, traj_res)

    if seg_call_num==1:
        curr_seg_info['larm_costs'] = (pos_errs, traj_res)
        

    ## then this is the last call to this logger, hence, dump the info to a file:
    if curr_seg >= num_segs:
        suturing_info = {'perturbations' : perturbations,
                         'recording_fname' : scene_recording_fname,
                         'segments_info' : segments_info}
        info_file = open(save_path, 'wb')
        cPickle.dump(suturing_info, info_file, protocol=2)
