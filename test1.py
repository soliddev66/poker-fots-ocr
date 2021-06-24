import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf


from module import Backbone_branch, Recognition_branch, RoI_rotate
from data_provider.data_utils import restore_rectangle, ground_truth_to_word

detect_part = Backbone_branch.Backbone(is_training=False)
roi_rotate_part = RoI_rotate.RoIRotate()
recognize_part = Recognition_branch.Recognition(is_training=False)
font = cv2.FONT_HERSHEY_SIMPLEX

def resize_image(im,
                 max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

def get_project_matrix_and_width(text_polyses,
                                 target_height=8.0):
    project_matrixes = []
    box_widths = []
    filter_box_masks = []
    # max_width = 0
    # max_width = 0

    for i in range(text_polyses.shape[0]):
        x1, y1, x2, y2, x3, y3, x4, y4 = text_polyses[i] / 4

        rotated_rect = cv2.minAreaRect(np.array([[x1, y1],
                                                 [x2, y2],
                                                 [x3, y3],
                                                 [x4, y4]],
                                                dtype=np.int64))
        box_w, box_h = rotated_rect[1][0], rotated_rect[1][1]

        if box_w <= box_h:
            box_w, box_h = box_h, box_w

        mapped_x1, mapped_y1 = (0, 0)
        mapped_x4, mapped_y4 = (0, 8)

        width_box = math.ceil(8 * box_w / box_h)
        width_box = int(min(width_box, 128)) # not to exceed feature map's width
        # width_box = int(min(width_box, 512)) # not to exceed feature map's width
        """
        if width_box > max_width:
            max_width = width_box
        """
        mapped_x2, mapped_y2 = (width_box, 0)
        # mapped_x3, mapped_y3 = (width_box, 8)

        src_pts = np.float32([(x1, y1), (x2, y2), (x4, y4)])
        dst_pts = np.float32([(mapped_x1, mapped_y1),
                              (mapped_x2, mapped_y2),
                              (mapped_x4, mapped_y4)])
        affine_matrix = cv2.getAffineTransform(dst_pts.astype(np.float32),
                                               src_pts.astype(np.float32))
        affine_matrix = affine_matrix.flatten()

        # project_matrix = cv2.getPerspectiveTransform(dst_pts.astype(np.float32), src_pts.astype(np.float32))
        # project_matrix = project_matrix.flatten()[:8]

        project_matrixes.append(affine_matrix)
        box_widths.append(width_box)

    project_matrixes = np.array(project_matrixes)
    box_widths = np.array(box_widths)

    return project_matrixes, box_widths

def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]

def recognize(frame, pos, size):
    # w = 800
    # h = 650
    # x = pos[0] + size[0] // 2 - w // 2
    # y = pos[1] + size[1] // 2 - h // 2
    #
    # if x < 0:
    #     x = 0
    # if y < 0:
    #     y = 0
    # if x + w > frame.shape[1]:
    #     x = frame.shape[1] - w
    # if y + h > frame.shape[0]:
    #     y = frame.shape[0] - h
    # frame = frame[y: y + h, x: x + w]
    #cv2.imshow('frame', frame1)
    #cv2.waitKey(-1)
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    im_resized, (ratio_h, ratio_w) = resize_image(im)

    timer = {'detect': 0, 'restore': 0, 'nms': 0, 'recog': 0}
    start = time.time()
    shared_feature_map, score, geometry = sess.run([shared_feature,
                                                    f_score,
                                                    f_geometry],
                                                   feed_dict={input_images: [im_resized]})
    boxes = []

    # cv2.rectangle(frame, (pos[0] - x, pos[1] - y), (pos[0] - x + size[0], pos[1] - y + size[1]), (0, 0, 255))
    # cv2.imshow('bb', frame)
    # cv2.waitKey(-1)
    box = [pos[0] * ratio_w, pos[1] * ratio_h, (pos[0] + size[0]) * ratio_w,
           pos[1] * ratio_h, (pos[0] + size[0]) * ratio_w, (pos[1] + size[1]) * ratio_h,
           pos[0] * ratio_w, (pos[1] + size[1]) * ratio_h, 0.5]

    boxes.append(box)
    boxes = np.asarray(boxes, dtype='f')
    timer['detect'] = time.time() - start
    start = time.time()  # reset for recognition
    if boxes is not None and boxes.shape[0] != 0:

        input_roi_boxes = boxes[:, :8].reshape(-1, 8)
        recog_decode_list = []
        # Here avoid too many text area leading to OOM
        for batch_index in range(input_roi_boxes.shape[0] // 32 + 1):  # test roi batch size is 32
            start_slice_index = batch_index * 32
            end_slice_index = (batch_index + 1) * 32 if input_roi_boxes.shape[0] >= (batch_index + 1) * 32 else \
                input_roi_boxes.shape[0]
            tmp_roi_boxes = input_roi_boxes[start_slice_index:end_slice_index]

            boxes_masks = [0] * tmp_roi_boxes.shape[0]
            transform_matrixes, box_widths = get_project_matrix_and_width(tmp_roi_boxes)

            # Run end to end
            recog_decode = sess.run(dense_decode,
                                    feed_dict={input_feature_map: shared_feature_map,
                                               input_transform_matrix: transform_matrixes,
                                               input_box_mask[0]: boxes_masks,
                                               input_box_widths: box_widths})
            recog_decode_list.extend([r for r in recog_decode])

        timer['recog'] = time.time() - start
        print(timer['recog'])
        # Preparing for draw boxes
        boxes = boxes[:, :8].reshape((-1, 4, 2))
        boxes[:, :, 0] /= ratio_w
        boxes[:, :, 1] /= ratio_h

        if len(recog_decode_list) != boxes.shape[0]:
            print("detection and recognition result are not equal!")
            exit(-1)

        box = boxes[0]
        # to avoid submitting errors
        box = sort_poly(box.astype(np.int32))
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            return None
        recognition_result = ground_truth_to_word(recog_decode_list[0])

        im_txt = cv2.putText(im[:, :, ::-1],
                             recognition_result,
                             (box[0, 0], box[0, 1]),
                             font,
                             0.5,
                             (0, 0, 255),
                             1)
        cv2.rectangle(im_txt, (box[0, 0], box[0, 1]), (box[2, 0], box[2, 1]), (0, 0, 255))
        cv2.imshow('demo', im_txt)
        cv2.waitKey(1)
        return recognition_result

    duration = time.time() - start_time
    print('[timing] {}'.format(duration))

if __name__ == '__main__':

    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"



    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32,
                                      shape=[None, None, None, 3],
                                      name='input_images')
        input_feature_map = tf.placeholder(tf.float32,
                                           shape=[None, None, None, 32],
                                           name='input_feature_map')
        input_transform_matrix = tf.placeholder(tf.float32,
                                                shape=[None, 6],
                                                name='input_transform_matrix')
        input_box_mask = []
        input_box_mask.append(tf.placeholder(tf.int32,
                                             shape=[None],
                                             name='input_box_masks_0'))
        input_box_widths = tf.placeholder(tf.int32,
                                          shape=[None],
                                          name='input_box_widths')

        input_seq_len = input_box_widths[tf.argmax(input_box_widths, 0)] * tf.ones_like(input_box_widths)
        global_step = tf.get_variable('global_step',
                                      [],
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        shared_feature, f_score, f_geometry = detect_part.model(input_images)
        pad_rois = roi_rotate_part.roi_rotate_tensor_pad(input_feature_map,
                                                         input_transform_matrix,
                                                         input_box_mask,
                                                         input_box_widths)
        recognition_logits = recognize_part.build_graph(pad_rois,
                                                        input_box_widths)
        _, dense_decode = recognize_part.decode(recognition_logits,
                                                input_box_widths)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        checkpoint_path = 'checkpoints/'
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
            model_path = os.path.join(checkpoint_path,
                                      os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)

            cap = cv2.VideoCapture('2021-06-09 19-24-22 - 34 - testing new version for the first time (4).mp4')
            crop_region = [[125, 32], [954, 650]]

            while (True):

                ret, frame = cap.read()
                if frame is None:
                    break
                frame = frame[crop_region[0][1]:crop_region[0][1] + crop_region[1][1],
                        crop_region[0][0]:crop_region[0][0] + crop_region[1][0]]
                cv2.imwrite('1.jpg', frame)
                #frame = cv2.imread('test_imgs/frame.jpg')
                pos = [763, 332]
                size = [86, 25]

                recognition_result = recognize(frame, pos, size)
                print(recognition_result)

