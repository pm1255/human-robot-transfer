"""Regression coverage for false targets found in actual visual inspection."""
import unittest
import cv2
import numpy as np
from human2robot.object_annotation.adaptive_segment import garment_proposal


class GarmentProposalTest(unittest.TestCase):
    def test_wall_and_worn_shirt_are_excluded(self):
        frame=np.full((480,640,3),155,np.uint8)  # gray wall
        frame[150:400]=[45,100,145]              # brown table
        frame[230:330,240:400]=200              # white garment
        frame[430:480,220:430]=210              # worn shirt below table
        box,roi,_=garment_proposal(frame,120,[],expand=True)
        self.assertIsNotNone(box)
        self.assertLess(box[3],410)
        self.assertGreater(box[1],200)
        self.assertFalse(roi[50,300])

    def test_out_of_view_is_missing(self):
        frame=np.full((480,640,3),150,np.uint8)
        box,_,_=garment_proposal(frame,3420,[],expand=True)
        self.assertIsNone(box)

    def test_connected_cloth_hanging_off_table_is_included(self):
        frame=np.full((480,640,3),50,np.uint8)
        frame[:100]=150
        frame[150:350]=[45,100,145]
        frame[240:410,240:400]=210
        box,_,_=garment_proposal(frame,240,[],expand=True)
        self.assertGreaterEqual(box[3],410)

if __name__=='__main__':unittest.main()
