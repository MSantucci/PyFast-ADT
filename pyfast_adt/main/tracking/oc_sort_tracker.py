# testing OC-sort algorithm for object tracking for pyfast adt

from ocsort import OCSort
#create instance of SORT
mot_tracker = OCSort(1)

# add a detection for every frame
# detections = [x1,y1,x2,y2,score] format. can be a list (usually is a list of multiple objects)
detections = [[10,10,20,20, 100]]
# update SORT
track_bbs_ids = mot_tracker.update(detections)

# track_bbs_ids is a np array where each row contains a valid bounding box and track_id (last column)
