import unittest
from compare_reviews import match_boundaries,compare
class ReviewTests(unittest.TestCase):
 def test_one_reference_cannot_match_twice(self):
  m=match_boundaries([1.,1.1],[1.05],.1);self.assertEqual(m['matches'],1);self.assertEqual(m['precision'],.5)
 def test_same_file_never_independent(self):
  d={'frames':10,'fps':10,'episode_id':'x','segments':[{'start_frame':0,'end_frame_exclusive':10,'phase':'transport','verb':'move'}]};r=compare(d,d);self.assertFalse(r['independent_review_evidence_present']);self.assertFalse(r['training_acceptance']);self.assertEqual(r['metrics']['object']['jointly_known_frames'],0)
 def test_gap_rejected(self):
  d={'frames':10,'fps':10,'segments':[{'start_frame':1,'end_frame_exclusive':10}]}
  with self.assertRaises(AssertionError):compare(d,d)
if __name__=='__main__':unittest.main()
