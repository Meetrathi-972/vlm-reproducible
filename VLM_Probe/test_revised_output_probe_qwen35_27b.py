import copy
import math
import unittest
from types import SimpleNamespace
import revised_output_probe_qwen35_27b as R
import evaluate_output_probe_qwen35_27b as E
from test_output_probe_qwen35_27b import obj,run

class RevisedTests(unittest.TestCase):
    def test_four_features_and_four_expected_signs(self):
        self.assertEqual(R.FEATURES,('free_space','clearance','regularity','spacing_consistency'))
        for signs in R.SIGN_MATRIX.values():
            self.assertEqual(set(signs),set(R.FEATURES))
            self.assertTrue(all(v in (-1,1) for v in signs.values()))
        a=run([obj('bed',0,0),obj('chair',3,0)],{'bed':1,'chair':1})
        for condition in R.SIGN_MATRIX:
            score=E.semantic(a,a,condition)
            self.assertEqual(score['direction_denominator'],4)
            self.assertEqual(score['direction_accuracy'],0)

    def test_missing_spacing_does_not_shrink_denominator(self):
        a=run([obj('bed',0,0),obj('chair',.2,0)],{'bed':1,'chair':1})
        s=E.semantic(a,a,'spacious')
        self.assertEqual(s['direction_denominator'],4)
        self.assertIsNone(s['direction_accuracy'])

    def test_facing_runs_original_verifier(self):
        a=run([obj('desk',0,0),obj('chair',0,2,anchor='desk',description='A chair facing the desk.',rotation=0)],{'desk':1,'chair':1})
        rows,score=R.relation_details(a)
        row=next(r for r in rows if r['relation']=='facing')
        self.assertEqual(row['result'],'SATISFIED')
        self.assertEqual(score['src'],1)
        a['scene']['objects'][1]['rotation_angle']=180
        self.assertEqual(R.relation_details(a)[0][0]['result'],'VIOLATED')
        verifier=R.orientation_verifier()
        self.assertEqual(verifier._atan2_to_yaw_convention(0,1),90)
        self.assertEqual(verifier._check_faces_point(SimpleNamespace(x=0,y=0,yaw=180),0,0),'pass')

    def test_front_and_facing_not_collapsed(self):
        a=run([obj('desk',0,0),obj('chair',0,2,anchor='desk',description='A chair in front of the desk, facing it.',rotation=0)],{'desk':1,'chair':1})
        rows,score=R.relation_details(a)
        self.assertEqual({r['relation'] for r in rows},{'front of','facing'})
        self.assertEqual(next(r for r in rows if r['relation']=='front of')['result'],'UNEVALUATED')
        self.assertEqual(score['evaluated_relations'],1)

    def test_vertical_intersection_is_not_fabricated_contact(self):
        a=run([obj('desk',0,0),obj('shelf',0,0,anchor='desk',description='A shelf on top of the desk.')],{'desk':1,'shelf':1})
        rows,score=R.relation_details(a)
        self.assertEqual(rows[0]['result'],'UNEVALUATED')
        self.assertIn('intersection=1',rows[0]['geometry_test'])
        self.assertIsNone(score['src'])
        self.assertEqual(score['evaluated_relations'],0)

    def test_cvr_details_expose_room_boundary_and_overlap(self):
        a=run([obj('bed',4.5,0),obj('chair',4.7,0)],{'bed':1,'chair':1})
        _,checks,_=E.validity(a)
        rows=R.violation_details(a,checks)
        self.assertEqual(sum(r['constraint']=='inside_room' for r in rows),2)
        self.assertTrue(any('x_max=5.5 exceeds room limit 5' in r['violation_reason'] for r in rows))
        self.assertTrue(any('intersection area=' in r['violation_reason'] for r in rows if r['constraint']=='footprint_nonoverlap'))

    def test_ccr_actual_type_change_and_movement(self):
        a=run([obj('bed',0,0)])
        b=copy.deepcopy(a);b['scene']['objects'][0]['position']=[1.5,.5];b['scene']['objects'][0]['bbox'][0]=1;b['scene']['objects'][0]['object_type']='ornaments'
        relevant,collateral=R.changes(a,b)
        self.assertEqual({r['channel'] for r in relevant},{'position','bbox'})
        self.assertEqual(collateral[0]['channel'],'object_type')
        self.assertEqual(collateral[0]['before'],'bulk')
        self.assertEqual(collateral[0]['after'],'ornaments')

if __name__=='__main__':unittest.main()
