import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2


from models import generator
from utils import DataLoader, load, save, psnr_error
from constant import const
import evaluate


slim = tf.contrib.slim

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
test_folder = const.TEST_FOLDER

num_his = const.NUM_HIS
height, width = 256, 256

snapshot_dir = const.SNAPSHOT_DIR
psnr_dir = const.PSNR_DIR
evaluate_name = const.EVALUATE

print(const)


# define dataset
with tf.name_scope('dataset'):
    test_video_clips_tensor = tf.placeholder(shape=[1, height, width, 3 * (num_his + 1)],
                                             dtype=tf.float32)
    test_inputs = test_video_clips_tensor[..., 0:num_his*3]
    test_gt = test_video_clips_tensor[..., -3:]
    print('test inputs = {}'.format(test_inputs))
    print('test prediction gt = {}'.format(test_gt))

# define testing generator function and
# in testing, only generator networks, there is no discriminator networks and flownet.
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_outputs = generator(test_inputs, layers=4, output_channel=3)
    test_psnr_error = psnr_error(gen_frames=test_outputs, gt_frames=test_gt)
    image_error = tf.abs((test_outputs - test_gt)*255)
    #image_error = cv2.resize(image_error,   
    


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    # dataset
    data_loader = DataLoader(test_folder, height, width)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt, dataset_name, evaluate_name):
        load(loader, sess, ckpt)

        psnr_records = []
        videos_info = data_loader.videos
        num_videos = len(videos_info.keys())
        total = 0
        timestamp = time.time()

        for video_name, video in videos_info.items():
            length = video['length']
            total += length
            psnrs = np.empty(shape=(length,), dtype=np.float32)

            for i in range(num_his, length):
                video_clip = data_loader.get_video_clips(video_name, i - num_his, i + 1)
                image_loss_out, psnr = sess.run((image_error, test_psnr_error),
                                feed_dict={test_video_clips_tensor: video_clip[np.newaxis, ...]})
                psnrs[i] = psnr

                print('video = {} / {}, i = {} / {}, psnr = {:.6f}'.format(
                    video_name, num_videos, i, length, psnr))
                    
                path_tmp_image = 'temp_image/' + str(video_name) + '/' + str(i) + '.jpg'
                os.makedirs(os.path.dirname(path_tmp_image), exist_ok=True)
                image_loss_out = np.uint8(image_loss_out)
                image_loss_out = image_loss_out.reshape(256, 256, 3)
                cv2.imwrite(path_tmp_image, image_loss_out)

            psnrs[0:num_his] = psnrs[num_his]
            psnr_records.append(psnrs)

        result_dict = {'dataset': dataset_name, 'psnr': psnr_records, 'flow': [], 'names': [], 'diff_mask': []}

        used_time = time.time() - timestamp
        print('total time = {}, fps = {}'.format(used_time, total / used_time))

        '''
        # TODO specify what's the actual name of ckpt.
        pickle_path = os.path.join(psnr_dir, os.path.split(ckpt)[-1])
        with open(pickle_path, 'wb') as writer:
            pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

        results = evaluate.evaluate(evaluate_name, pickle_path)
        print(results)
        '''


    inference_func(snapshot_dir, dataset_name, evaluate_name)
