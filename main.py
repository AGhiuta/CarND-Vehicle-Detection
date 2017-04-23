import os
import pickle

from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

from utils import *

orient = 7  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
hist_range = (0, 256)
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
x_start_stop = [700, None]
svc, X_scaler = None, None

def process_frame(image):
	draw_image = np.copy(image)

	overlap = 0.9
	windows = []

	windows += slide_window(image, x_start_stop=x_start_stop,
		y_start_stop=[380, 560], xy_window=(120, 96), xy_overlap=(overlap, overlap))

	windows += slide_window(image, x_start_stop=x_start_stop,
		y_start_stop=[380, 650], xy_window=(280, 224), xy_overlap=(overlap, overlap))

	hot_windows = search_windows(image, windows, svc, X_scaler, conv='RGB2YCrCb', 
                    spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range,
                    orient=orient, pix_per_cell=pix_per_cell, 
                    cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                    hist_feat=hist_feat, hog_feat=hog_feat)

	heat = np.zeros_like(image[:,:,0]).astype(np.float)
	heat = add_heat(heat, hot_windows)

	# Apply threshold to help remove false positives
	heat = apply_threshold(heat, 2)

	# Visualize the heatmap when displaying    
	heatmap = np.clip(heat, 0, 255)

	# Find final boxes from heatmap using label function
	labels = label(heatmap)
	window_img = draw_labeled_bboxes(draw_image, labels)

	return window_img

def main():
	global svc, X_scaler

	if not os.path.exists('./svc.pkl'):
		svc, X_scaler = train_model(conv='BGR2YCrCb', spatial_size=spatial_size,
				hist_bins=hist_bins, hist_range=hist_range, orient=orient,
				pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
				hog_channel=hog_channel,
				spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)
	else:
		with open('./svc.pkl', 'rb') as f:
			svc = pickle.load(f)
		with open('./scaler.pkl', 'rb') as f:
			X_scaler = pickle.load(f)

	video = 'project_video'
	video_out = '{}_out.mp4'.format(video)
	clip1 = VideoFileClip('{}.mp4'.format(video))

	out_clip = clip1.fl_image(process_frame)
	out_clip.write_videofile(video_out, audio=False)



if __name__ == '__main__':
	main()