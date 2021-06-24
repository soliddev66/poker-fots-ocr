import threading
import pyautogui
from datetime import date, datetime
from time import sleep
import re

import cv2
import time
import math
import os
import numpy as np
import tensorflow as tf

from module import Backbone_branch, Recognition_branch, RoI_rotate
from data_provider.data_utils import restore_rectangle, ground_truth_to_word

class Tracker(threading.Thread):

    def __init__(self, *args, **kwargs):

        # init ocr engine
        self.cap = cv2.VideoCapture('2021-06-09 19-24-22 - 34 - testing new version for the first time (2).mp4')
        self.writeLogTime()
        os.environ['CUDA_VISIBLE_DEVICES'] = "1"
        self.crop_region = [[125,32],[954,650]]
        self.pos_names = ['BB', 'EP', 'MP', 'CO', 'BU', 'SB']
        self.g_betsz = 1 # current betsize
        self.g_bb_pos = 0 # current big blind position
        self.g_potsz = 0 # current pot size
        self.g_prev_potsz = 0 # previous pot size
        self.g_action_pos = 0 # the postion of the player that will do the next action
        self.g_bet_sizes = [0, 0, 0, 0, 0, 0]
        self.g_current_street = -1 # current street (0: preflop 1: flop 2: turn 3: river)
        self.g_community_cards = ''
        self.tmp_community_cards = ''

        self.flop_flag = False
        self.turn_flag = False
        self.river_flag = False
        #
        self.detect_part = Backbone_branch.Backbone(is_training=False)
        self.roi_rotate_part = RoI_rotate.RoIRotate()
        self.recognize_part = Recognition_branch.Recognition(is_training=False)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        #
        super(Tracker, self).__init__(*args, **kwargs)
        self.__flag = threading.Event() # The flag used to pause the thread
        self.__flag.set() # Set to True
        self.__running = threading.Event() # Used to stop the thread identification
        self.__running.set() # Set running to True

    def run(self):
        with tf.get_default_graph().as_default():
            self.input_images = tf.placeholder(tf.float32,
                                          shape=[None, None, None, 3],
                                          name='input_images')
            self.input_feature_map = tf.placeholder(tf.float32,
                                               shape=[None, None, None, 32],
                                               name='input_feature_map')
            self.input_transform_matrix = tf.placeholder(tf.float32,
                                                    shape=[None, 6],
                                                    name='input_transform_matrix')
            self.input_box_mask = []
            self.input_box_mask.append(tf.placeholder(tf.int32,
                                                 shape=[None],
                                                 name='input_box_masks_0'))
            self.input_box_widths = tf.placeholder(tf.int32,
                                              shape=[None],
                                              name='input_box_widths')

            input_seq_len = self.input_box_widths[tf.argmax(self.input_box_widths, 0)] * tf.ones_like(self.input_box_widths)
            global_step = tf.get_variable('global_step',
                                          [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)

            self.shared_feature, self.f_score, self.f_geometry = self.detect_part.model(self.input_images)
            pad_rois = self.roi_rotate_part.roi_rotate_tensor_pad(self.input_feature_map,
                                                             self.input_transform_matrix,
                                                             self.input_box_mask,
                                                             self.input_box_widths)
            recognition_logits = self.recognize_part.build_graph(pad_rois,
                                                            self.input_box_widths)
            _, self.dense_decode = self.recognize_part.decode(recognition_logits,
                                                    self.input_box_widths)

            variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
            saver = tf.train.Saver(variable_averages.variables_to_restore())

            checkpoint_path = 'checkpoints/'
            with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                self.sess = sess
                ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
                model_path = os.path.join(checkpoint_path,
                                          os.path.basename(ckpt_state.model_checkpoint_path))
                print('Restore from {}'.format(model_path))
                saver.restore(sess, model_path)
                while self.__running.isSet():
                    self.__flag.wait() # return immediately when it is True, block until the internal flag is True when it is False
                    #print(time.time())
                    #img = pyautogui.screenshot()
                    #frame = np.array(img)
                    #frame = frame[:, :, ::-1].copy()

                    ret, frame = self.cap.read()
                    if frame is None:
                        continue

                    # get the frame of the specific region
                    frame = frame[self.crop_region[0][1]:self.crop_region[0][1] + self.crop_region[1][1], self.crop_region[0][0]:self.crop_region[0][0] + self.crop_region[1][0]]
                    #cv2.imshow('frame',frame)
                    #cv2.waitKey(1)

                    self.g_potsz = self.getPotSize(frame)

                    if self.g_potsz == None:
                        self.g_current_street = -1
                        continue
                    # initial frame of preflop
                    if self.g_potsz == 1.5 and self.g_current_street == -1:
                        holecards = self.getHoleCards(frame)
                        if len(holecards) != 4:
                            continue
                        self.g_bb_pos = self.getBigBlindPos(frame, None)
                        self.g_bet_sizes = [0, 0, 0, 0, 0, 0]
                        self.g_bet_sizes[self.g_bb_pos] = 1
                        self.g_bet_sizes[(self.g_bb_pos + 5) % 6] = 0.5
                        self.g_prev_potsz = 1.5
                        self.g_betsz = 1
                        self.g_action_pos = (self.g_bb_pos + 1) % 6
                        stacksz_list = self.getStackSizeList(frame)
                        # Log inital stack size
                        f = open("log.txt", "a")
                        f.write("\nBB: ($%s in chips)\n" % (stacksz_list[(self.g_bb_pos) % 6] + 1))
                        f.write("SB: ($%s in chips)\n" % (stacksz_list[(self.g_bb_pos + 5) % 6] + 0.5))
                        f.write("BU: ($%s in chips)\n" % (stacksz_list[(self.g_bb_pos + 4) % 6]))
                        f.write("CO: ($%s in chips)\n" % (stacksz_list[(self.g_bb_pos + 3) % 6]))
                        f.write("MP: ($%s in chips)\n" % (stacksz_list[(self.g_bb_pos + 2) % 6]))
                        f.write("EP: ($%s in chips)\n" % (stacksz_list[(self.g_bb_pos + 1) % 6]))
                        f.write("BB: posts big blind $1\nSB: posts small blind $0.5\n")
                        f.write("*** HOLE CARDS ***\n")
                        f.write("Dealt to Hero [%s]\n" % (holecards))
                        f.close()

                        print("\nBB: ($%s in chips)" % (stacksz_list[(self.g_bb_pos) % 6] + 1))
                        print("SB: ($%s in chips)" % (stacksz_list[(self.g_bb_pos + 5) % 6] + 0.5))
                        print("BU: ($%s in chips)" % (stacksz_list[(self.g_bb_pos + 4) % 6]))
                        print("CO: ($%s in chips)" % (stacksz_list[(self.g_bb_pos + 3) % 6]))
                        print("MP: ($%s in chips)" % (stacksz_list[(self.g_bb_pos + 2) % 6]))
                        print("EP: ($%s in chips)" % (stacksz_list[(self.g_bb_pos + 1) % 6]))
                        print("BB: posts big blind $1\nSB: posts small blind $0.5")
                        print("*** HOLE CARDS ***")
                        print("Dealt to Hero [%s]" % (holecards))

                        self.g_current_street = 0
                        self.g_community_cards = ''
                        self.flop_flag = False
                        self.turn_flag = False
                        self.river_flag = False

                    # preflop range
                    elif self.g_current_street == 0:
                        actionType = self.getActionType(frame)
                        if actionType == None:
                            continue
                        elif actionType == 'Raise':
                            f = open("log.txt", "a")
                            f.write("%s: raises to $%.2f\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                            f.close()
                            print("%s: raises to $%.2f" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                            next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                            if next_street_start == -1:
                                self.g_action_pos = next_action_pos
                            elif next_street_start == 0:
                                self.g_current_street = -1
                            else:
                                self.g_current_street += 1
                        elif actionType == 'Fold':
                            f = open("log.txt", "a")
                            f.write("%s: folds\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6]))
                            f.close()
                            print("%s: folds" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6]))
                            next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                            if next_street_start == -1:
                                self.g_action_pos = next_action_pos
                            elif next_street_start == 0:
                                self.g_current_street = -1
                            else:
                                self.g_current_street += 1
                        elif actionType == 'Call':
                            f = open("log.txt", "a")
                            f.write("%s: calls $%.2f\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                            f.close()
                            print("%s: calls $%.2f" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                            next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                            if next_street_start == -1:
                                self.g_action_pos = next_action_pos
                            elif next_street_start == 0:
                                self.g_current_street = -1
                            else:
                                self.g_current_street += 1
                        elif actionType == 'Check':
                            f = open("log.txt", "a")
                            f.write("%s: checks\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6]))
                            f.close()
                            print("%s: checks" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6]))
                            next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                            if next_street_start == -1:
                                self.g_action_pos = next_action_pos
                            elif next_street_start == 0:
                                self.g_current_street = -1
                            else:
                                self.g_current_street += 1
                    # # flop range
                    # elif self.g_current_street == 1:
                    #     if self.flop_flag == False:
                    #         for i in range(len(self.g_bet_sizes)):
                    #             if self.g_bet_sizes[i] > -1:
                    #                 self.g_bet_sizes[i] = 0
                    #         state = self.getCommunityCards(frame, self.g_current_street)
                    #         if state == False:
                    #             continue
                    #         f =open("log.txt", "a")
                    #         f.write("*** FLOP *** %s\n" % (self.g_community_cards))
                    #         f.close()
                    #         print("*** FLOP *** %s" % (self.g_community_cards))
                    #         self.g_action_pos = (self.g_bb_pos + 4) % 6
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #         self.flop_flag = True

                    #     actionType = self.getActionType(frame)
                    #     if actionType == None:
                    #         continue
                    #     elif actionType == 'Raise':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: raises to $%.2f\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         f.close()
                    #         print("%s: raises to $%.2f" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #     elif actionType == 'Fold':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: folds\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6]))
                    #         f.close()
                    #         print("%s: folds" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6]))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #     elif actionType == 'Call':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: calls $%.2f\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         f.close()
                    #         print("%s: calls $%.2f" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #     elif actionType == 'Check':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: checks\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6]))
                    #         f.close()
                    #         print("%s: checks" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6]))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #     elif actionType == 'Bet':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: bets $%.2f\n" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         f.close()
                    #         print("%s: bets $%.2f" % (self.pos_names[(self.g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    # # turn
                    # elif self.g_current_street == 2:
                    #     if self.turn_flag == False:
                    #         for i in range(len(self.g_bet_sizes)):
                    #             if self.g_bet_sizes[i] > -1:
                    #                 self.g_bet_sizes[i] = 0
                    #         state = self.getCommunityCards(frame, self.g_current_street)
                    #         if state == False:
                    #             continue
                    #         f =open("log.txt", "a")
                    #         f.write("*** TURN *** %s\n" % (self.g_community_cards))
                    #         f.close()
                    #         print("*** TURN *** %s" % (self.g_community_cards))
                    #         g_action_pos = (self.g_bb_pos + 4) % 6
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #         self.turn_flag = True

                    #     actionType = self.getActionType(frame)
                    #     if actionType == None:
                    #         continue
                    #     elif actionType == 'Raise':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: raises to $%.2f\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         f.close()
                    #         print("%s: raises to $%.2f" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #     elif actionType == 'Fold':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: folds\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6]))
                    #         f.close()
                    #         print("%s: folds" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6]))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #     elif actionType == 'Call':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: calls $%.2f\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         f.close()
                    #         print("%s: calls $%.2f" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #     elif actionType == 'Check':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: checks\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6]))
                    #         f.close()
                    #         print("%s: checks" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6]))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    #     elif actionType == 'Bet':
                    #         f = open("log.txt", "a")
                    #         f.write("%s: bets $%.2f\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         f.close()
                    #         print("%s: bets $%.2f" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                    #         next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                    #         if next_street_start == -1:
                    #             self.g_action_pos = next_action_pos
                    #         elif next_street_start == 0:
                    #             self.g_current_street = -1
                    #             continue
                    #         else:
                    #             self.g_current_street += 1
                    #             continue
                    # # river
                    # elif self.g_current_street == 3:
                        # if river_flag == False:
                        #     for i in range(len(self.g_bet_sizes)):
                        #         if self.g_bet_sizes[i] > -1:
                        #             self.g_bet_sizes[i] = 0
                        #     self.tmp_community_cards = self.g_community_cards
                        #     state = self.getCommunityCards(frame, self.g_current_street)
                        #     if state == False:
                        #         continue
                        #     if len(self.g_community_cards) != 18:
                        #         self.g_community_cards = self.tmp_community_cards
                        #         continue
                        #     f =open("log.txt", "a")
                        #     f.write("*** RIVER *** %s\n" % (self.g_community_cards))
                        #     f.close()
                        #     print("*** RIVER *** %s" % (self.g_community_cards))
                        #     self.g_action_pos = (self.g_bb_pos + 4) % 6
                        #     next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                        #     if next_street_start == -1:
                        #         self.g_action_pos = next_action_pos
                        #     elif next_street_start == 0:
                        #         self.g_current_street = -1
                        #         continue
                        #     else:
                        #         self.g_current_street = -1
                        #         continue
                        #     river_flag = True

                        # actionType = self.getActionType(frame)
                        # if actionType == None:
                        #     continue
                        # elif actionType == 'Raise':
                        #     f = open("log.txt", "a")
                        #     f.write("%s: raises to $%.2f\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                        #     f.close()
                        #     print("%s: raises to $%.2f" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                        #     next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                        #     if next_street_start == -1:
                        #         self.g_action_pos = next_action_pos
                        #     elif next_street_start == 0:
                        #         self.g_current_street = -1
                        #         continue
                        #     else:
                        #         self.g_current_street = -1
                        #         continue
                        # elif actionType == 'Fold':
                        #     f = open("log.txt", "a")
                        #     f.write("%s: folds\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6]))
                        #     f.close()
                        #     print("%s: folds" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6]))
                        #     next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                        #     if next_street_start == -1:
                        #         self.g_action_pos = next_action_pos
                        #     elif next_street_start == 0:
                        #         self.g_current_street = -1
                        #         continue
                        #     else:
                        #         self.g_current_street = -1
                        #         continue
                        # elif actionType == 'Call':
                        #     f = open("log.txt", "a")
                        #     f.write("%s: calls $%.2f\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                        #     f.close()
                        #     print("%s: calls $%.2f" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                        #     next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                        #     if next_street_start == -1:
                        #         self.g_action_pos = next_action_pos
                        #     elif next_street_start == 0:
                        #         self.g_current_street = -1
                        #         continue
                        #     else:
                        #         self.g_current_street = -1
                        #         continue
                        # elif actionType == 'Check':
                        #     f = open("log.txt", "a")
                        #     f.write("%s: checks\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6]))
                        #     f.close()
                        #     print("%s: checks" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6]))
                        #     next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                        #     if next_street_start == -1:
                        #         self.g_action_pos = next_action_pos
                        #     elif next_street_start == 0:
                        #         self.g_current_street = -1
                        #         continue
                        #     else:
                        #         self.g_current_street = -1
                        #         continue
                        # elif actionType == 'Bet':
                        #     f = open("log.txt", "a")
                        #     f.write("%s: bets $%.2f\n" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                        #     f.close()
                        #     print("%s: bets $%.2f" % (self.pos_names[(g_action_pos - self.g_bb_pos + 6) % 6], self.g_betsz))
                        #     next_action_pos, next_street_start = self.getNextPlayerandStreetStartStatus()
                        #     if next_street_start == -1:
                        #         self.g_action_pos = next_action_pos
                        #     elif next_street_start == 0:
                        #         self.g_current_street = -1
                        #         continue
                        #     else:
                        #         self.g_current_street = -1
                        #         continue

                    sleep(0.01)
    def pause(self):
        self.__flag.clear() # Set to False to block the thread

    def resume(self):
        self.__flag.set() # Set to True, let the thread stop blocking

    def stop(self):
        self.__flag.set() # Resume the thread from the suspended state, if it is already suspended
        self.__running.clear() # Set to False

    ###### write current time in the log file
    def writeLogTime(self):
        now = datetime.now()
        dt_string = now.strftime("%Y/%m/%d %H:%M:%S CET\n")
        f = open("log.txt", "w")
        f.write(dt_string)
        f.close()

    ####### get pot size
    ## param: frame
    ## output: return the pot size if recognition is successful, return None if the recognition fails.

    def getPotSize(self, frame):
        potsz_size = [50,24]
        potsz_pos = [482, 200]
        potsize_string = self.recognizeSize(frame, potsz_pos, potsz_size)
        try:
            potsize = float(potsize_string)
            return potsize
        except ValueError:
            return None

    ###### get the action type of the next player
    ## param: frame
    ## output: the action of the next player

    def getActionType(self, frame):
        scrnamebox_list =[[787,131],[790,334],[469,424],[110,334],[109,130],[434,91]]
        scrnamebox_size = [44,19]
        betbox_list =[[585,165],[605,326],[408,339],[263,324],[267,187],[420,163]]    #337
        betbox_size = [79,24]
        #betbox_me = [416,317]
        #betbox_size_me = [125,59]

        action_string = self.recognize(frame, scrnamebox_list[self.g_action_pos], scrnamebox_size)
        action_string = action_string.lower()
        if action_string == 'fold':
            self.g_bet_sizes[self.g_action_pos] = -1
            #self.g_prev_potsz = self.g_potsz
            return 'Fold'
        if action_string == 'check':
            self.g_bet_sizes[self.g_action_pos] = 0
            #self.g_prev_potsz = self.g_potsz
            self.g_betsz = 0
            return 'Check'
        
        flag = False
        betsz_string = self.recognizeSize(frame, betbox_list[self.g_action_pos], betbox_size)
        # if the action palyer is the 'hero'
        if self.g_action_pos == 2:
            #betsz_string = self.recognize(frame, betbox_me, betbox_size_me)
            try:
                betsz = float(betsz_string)
                flag = True
                if betsz > self.g_betsz:
                    action_string = 'raise'
                elif betsz == self.g_betsz:
                    action_string = 'call'
                else:
                    return None
            except ValueError:
                betsz = self.g_potsz - self.g_prev_potsz
                if betsz > self.g_betsz:
                    action_string = 'raise'
                elif betsz == self.g_betsz:
                    action_string = 'call'
                else:
                    return None
        # if the action player is not the 'hero'
        
        try:
            betsz = float(betsz_string)
            flag = True
        except ValueError:
            betsz = self.g_potsz - self.g_prev_potsz

        if action_string == 'raise' and betsz > self.g_betsz:
            if flag == True:
                self.g_bet_sizes[self.g_action_pos] = betsz
            else:
                self.g_bet_sizes[self.g_action_pos] += self.g_potsz - self.g_prev_potsz
            self.g_prev_potsz = self.g_potsz
            self.g_betsz = betsz
            return 'Raise'
        elif action_string == 'call' and self.g_potsz > self.g_prev_potsz:
            if flag == True:
                self.g_bet_sizes[self.g_action_pos] = betsz
            else:
                self.g_bet_sizes[self.g_action_pos] += self.g_potsz - self.g_prev_potsz
            self.g_prev_potsz = self.g_potsz
            return 'Call'
        elif (action_string == 'bet' or action_string == 'b') and self.g_potsz > self.g_prev_potsz:
            if flag == True:
                self.g_bet_sizes[self.g_action_pos] = betsz
            else:
                self.g_bet_sizes[self.g_action_pos] += self.g_potsz - self.g_prev_potsz
            self.g_prev_potsz = self.g_potsz
            self.g_betsz = betsz
            return 'Bet'
        else:
            return None

    ###### get the action type of the next player
    ## param: frame
    ## output: the action of the next player and check if the next street is ready to go
    ##                            if nextStreetStart is equal to zero it means to start the preflop again,
    ##                            if nextStreetStart is equal to one it means to start the next street

    def getNextPlayerandStreetStartStatus(self):
        nextStreetStart = -1
        next_action_pos = -1
        count = 0

        for i in range(1, 7):
            if self.g_bet_sizes[(self.g_action_pos + i) % 6] > -1:
                if count == 0:
                    next_action_pos = (self.g_action_pos + i) % 6
                    count += 1
                else:
                    count += 1
        if next_action_pos == -1 or count == 1:
            nextStreetStart = 0
            return next_action_pos, nextStreetStart
        else:
            flag = False
            for i in range(1, 6):
                if self.g_bet_sizes[next_action_pos] != self.g_bet_sizes[(next_action_pos + i) % 6] and self.g_bet_sizes[(next_action_pos + i) % 6] > -1:
                    flag = True
                    break

            if flag == False and self.g_bet_sizes[next_action_pos] > 0:
                nextStreetStart = 1
            return next_action_pos, nextStreetStart

    ####### get community cards
    ## param: frame and street number
    ## output: state:             return true if next street start, other wise return false
    ##                 card_string: return the string that represents community cards

    def getCommunityCards(self, frame, streetNo):
        card_pos = [[322,229],[385,229],[449,229],[513,229],[576,229]]
        card_size = [29,35]
        state = True
        full_string="123456789TJQK"
        if streetNo == 1:
            for i in range(1, 4):
                card_string = self.recognize(frame, card_pos[i - 1], card_size)
                card_string = 'T' if card_string == '10' else card_string
                if full_string.find(card_string) == -1 or card_string == '':
                    state = False
                    return state
                card_suite = self.getSuite(frame, card_pos[i - 1][0] + card_size[0], card_pos[i - 1][1] + card_size[1])
                self.g_community_cards += card_string + card_suite
            self.g_community_cards = '[' + self.g_community_cards + ']'
        elif streetNo == 2:
            card_string = self.recognize(frame, card_pos[3], card_size)
            card_string = 'T' if card_string == '10' else card_string
            if full_string.find(card_string) == -1 or card_string == '':
                state = False
                return state
            card_suite = self.getSuite(frame, card_pos[3][0] + card_size[0], card_pos[3][1] + card_size[1])
            self.g_community_cards += ' [' + card_string + card_suite + ']'
        elif streetNo == 3:
            card_string = self.recognize(frame, card_pos[4], card_size)
            card_string = 'T' if card_string == '10' else card_string
            if full_string.find(card_string) == -1 or card_string == '':
                state = False
                return state
            card_suite = self.getSuite(frame, card_pos[4][0] + card_size[0], card_pos[4][1] + card_size[1])
            self.g_community_cards += ' [' + card_string + card_suite + ']'
        return state

    ####### get the suite of the card
    ## param: frame, the position of the card
    ## output: return the suite of the card

    def getSuite(self, frame, x, y):
        if abs(int(frame[y, x, 0]) - int(frame[y, x, 1])) < 10 and abs(int(frame[y, x, 0]) - int(frame[y, x, 2])) < 10:
            return 's'
        if frame[y, x, 0] > 100 and frame[y, x, 1] > 100 and frame[y, x, 2] < 100:
            return 'd'
        elif frame[y, x, 0] < 100 and frame[y, x, 1] > 100 and frame[y, x, 2] > 100:
            return 'c'
        elif frame[y, x, 0] < 100 and frame[y, x, 1] < 100 and frame[y, x, 2] > 100:
            return 'h'
        else:
            return 'x'

    ####### get the stack sizes of all players
    ## param: frame
    ## output: return the stack sizes of all players (stack size -1 means the failure of recognition)

    def getStackSizeList(self, frame):
        p = '[\d]+[.,\d]+|[\d]*[.][\d]+|[\d]+'

        stcksz_pos_list = [[770,149],[769,353],[454,443],[80,353],[80,149],[403,106]]
        stcksz_size = [91,25]
        res = []
        for stcksz_pos in stcksz_pos_list:
            stacksz_string = self.recognizeSize(frame, stcksz_pos, stcksz_size)
            #res.append(stacksz_string)

            if re.search(p, stacksz_string) is not None:
                for catch in re.finditer(p, stacksz_string):
                    res.append(float(catch[0]))
                    break
            else:
                res.append(0)
        return res

    ####### get the position of big blind
    ## param: frame, current pos
    ## output: return the current postion of big blind

    def getBigBlindPos(self, frame, curr_pos):
        dial_btn_pos = [[732,207],[655,364],[386,369],[234,297],[218,213],[518,176]]
        if curr_pos == None:
            for i in range(len(dial_btn_pos)):
                if frame[dial_btn_pos[i][1], dial_btn_pos[i][0], 2] > 50:
                    curr_pos = (i + 2) % 6
                    break
        else:
            if frame[dial_btn_pos[(curr_pos + 1) % 6][1], dial_btn_pos[(curr_pos + 1) % 6][0], 2] > 50:
                curr_pos = (curr_pos + 1) % 6
        return curr_pos

    ####### recognize the hole cards of the hero
    ## param: frame
    ## output: return the hole cards

    def getHoleCards(self, frame):
        hero_pos = [[416,382],[477,383]]
        hero_size = [29,35]
        first_card_string = self.recognize(frame, hero_pos[0], hero_size)
        second_card_string = self.recognize(frame, hero_pos[1], hero_size)
        first_card_suite = self.getSuite(frame, hero_pos[0][0] + hero_size[0], hero_pos[0][1] + hero_size[1])
        second_card_suite = self.getSuite(frame, hero_pos[1][0] + hero_size[0], hero_pos[1][1] + hero_size[1])
        if first_card_string == '10':
            first_card_string = 'T'
        elif first_card_string == 'a':
            first_card_string = 'Q'
        elif first_card_string == 'G':
            first_card_string = 'Q'
        if second_card_string == '10':
            second_card_string = 'T'
        elif second_card_string == 'a':
            second_card_string = 'Q'
        elif second_card_string == 'G':
            second_card_string = 'Q'
    
        first_char = '0'
        second_char = '0'
        if first_card_string == 'T':
            first_char = 'I'
        elif first_card_string == 'K':
            first_char = 'R'
        elif first_card_string == 'A':
            first_char = 'S'
        else:
            first_char = first_card_string

        if second_card_string == '10':
            second_char = 'I'
        elif second_card_string == 'K':
            second_char = 'R'
        elif second_card_string == 'A':
            second_char = 'S'
        else:
            second_char = second_card_string

        if first_char > second_char:
            return first_card_string + first_card_suite + second_card_string + second_card_suite
        else:
            return second_card_string + second_card_suite + first_card_string + first_card_suite

    def resize_image(self, im, max_side_len=2400):
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

    def get_project_matrix_and_width(self, text_polyses,
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
            width_box = int(min(width_box, 128))  # not to exceed feature map's width
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

    def sort_poly(self, p):
        min_axis = np.argmin(np.sum(p, axis=1))
        p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
        if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
            return p
        else:
            return p[[0, 3, 2, 1]]

    def recognize(self, frame, pos, size):
        crop_img = frame[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]]
        # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        # ret, crop_img = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #cv2.imshow('crop', crop_img)

        lower = np.array([200, 200, 200])
        upper = np.array([255, 255, 255])
        mask = cv2.inRange(crop_img, lower, upper)

        # cv2.imshow('cc', mask)
        # cv2.imwrite('cc.jpg', mask)
        # cv2.waitKey(1)

        x_values = np.where(np.sum(mask, axis=0) > 0)[0]
        #print(x_values)
        flag = False
        pos_ = 0
        size_ = 0
        if np.size(x_values) != 0:
            x_start = x_values[0]
            #print(x_start)
            x_end = x_values[-1]
            #print(x_end)
            #print('--------------')
            pos_ = pos[0] + x_start - 4
            size_ = x_end - x_start + 8
            if (size_ > 5):
                flag = True
            else:
                flag = False
        else:
            flag = False

        if flag == True:
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            im_resized, (ratio_h, ratio_w) = self.resize_image(im)

            timer = {'detect': 0, 'restore': 0, 'nms': 0, 'recog': 0}
            start = time.time()
            shared_feature_map, score, geometry = self.sess.run([self.shared_feature,
                                                            self.f_score,
                                                            self.f_geometry],
                                                           feed_dict={self.input_images: [im_resized]})
            boxes = []

            # cv2.rectangle(frame, (pos[0] - x, pos[1] - y), (pos[0] - x + size[0], pos[1] - y + size[1]), (0, 0, 255))
            # cv2.imshow('bb', frame)
            # cv2.waitKey(-1)
            box = [pos_ * ratio_w, pos[1] * ratio_h, (pos_ + size_) * ratio_w,
                   pos[1] * ratio_h, (pos_ + size_) * ratio_w, (pos[1] + size[1]) * ratio_h,
                   pos_ * ratio_w, (pos[1] + size[1]) * ratio_h, 0.5]

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
                    transform_matrixes, box_widths = self.get_project_matrix_and_width(tmp_roi_boxes)

                    # Run end to end
                    recog_decode = self.sess.run(self.dense_decode,
                                            feed_dict={self.input_feature_map: shared_feature_map,
                                                       self.input_transform_matrix: transform_matrixes,
                                                       self.input_box_mask[0]: boxes_masks,
                                                       self.input_box_widths: box_widths})
                    recog_decode_list.extend([r for r in recog_decode])

                timer['recog'] = time.time() - start
                #print(timer['recog'])
                # Preparing for draw boxes
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

                if len(recog_decode_list) != boxes.shape[0]:
                    print("detection and recognition result are not equal!")
                    exit(-1)

                box = boxes[0]
                # to avoid submitting errors
                box = self.sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    return None
                recognition_result = ground_truth_to_word(recog_decode_list[0])

                im_txt = cv2.putText(im[:, :, ::-1],
                                     recognition_result,
                                     (box[0, 0], box[0, 1]),
                                     self.font,
                                     0.5,
                                     (0, 0, 255),
                                     1)
                cv2.rectangle(im_txt, (box[0, 0], box[0, 1]), (box[2, 0], box[2, 1]), (0, 0, 255))
                cv2.imshow('demo', im_txt)
                cv2.waitKey(1)
                return recognition_result
        else:
            return ''

    def recognizeSize(self, frame, pos, size):
        crop_img = frame[pos[1]:pos[1] + size[1], pos[0]:pos[0] + size[0]]
        crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        ret, crop_img = cv2.threshold(crop_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        x_values = np.where(np.sum(crop_img, axis=0) > 250)[0]

        flag = False
        pos_ = 0
        size_ = 0
        if np.size(x_values) != 0:
            x_start = x_values[0]
            x_end = x_values[-1]
            pos_ = pos[0] + x_start - 4
            size_ = x_end - x_start + 8
            if (size_ > 5):
                flag = True
            else:
                flag = False
        else:
            flag = False

        if flag == True:
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            start_time = time.time()
            im_resized, (ratio_h, ratio_w) = self.resize_image(im)

            timer = {'detect': 0, 'restore': 0, 'nms': 0, 'recog': 0}
            start = time.time()
            shared_feature_map, score, geometry = self.sess.run([self.shared_feature,
                                                                 self.f_score,
                                                                 self.f_geometry],
                                                                feed_dict={self.input_images: [im_resized]})
            boxes = []

            # cv2.rectangle(frame, (pos[0] - x, pos[1] - y), (pos[0] - x + size[0], pos[1] - y + size[1]), (0, 0, 255))
            # cv2.imshow('bb', frame)
            # cv2.waitKey(-1)
            box = [pos_ * ratio_w, pos[1] * ratio_h, (pos_ + size_) * ratio_w,
                   pos[1] * ratio_h, (pos_ + size_) * ratio_w, (pos[1] + size[1]) * ratio_h,
                   pos_ * ratio_w, (pos[1] + size[1]) * ratio_h, 0.5]

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
                    transform_matrixes, box_widths = self.get_project_matrix_and_width(tmp_roi_boxes)

                    # Run end to end
                    recog_decode = self.sess.run(self.dense_decode,
                                                 feed_dict={self.input_feature_map: shared_feature_map,
                                                            self.input_transform_matrix: transform_matrixes,
                                                            self.input_box_mask[0]: boxes_masks,
                                                            self.input_box_widths: box_widths})
                    recog_decode_list.extend([r for r in recog_decode])

                timer['recog'] = time.time() - start
                #print(timer['recog'])
                # Preparing for draw boxes
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                boxes[:, :, 0] /= ratio_w
                boxes[:, :, 1] /= ratio_h

                if len(recog_decode_list) != boxes.shape[0]:
                    print("detection and recognition result are not equal!")
                    exit(-1)

                box = boxes[0]
                # to avoid submitting errors
                box = self.sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    return None
                recognition_result = ground_truth_to_word(recog_decode_list[0])

                im_txt = cv2.putText(im[:, :, ::-1],
                                     recognition_result,
                                     (box[0, 0], box[0, 1]),
                                     self.font,
                                     0.5,
                                     (0, 0, 255),
                                     1)
                cv2.rectangle(im_txt, (box[0, 0], box[0, 1]), (box[2, 0], box[2, 1]), (0, 0, 255))
                cv2.imshow('demo', im_txt)
                cv2.waitKey(1)
                return recognition_result
            else:
                return ''

        
