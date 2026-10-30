"""Offline scientific-regression checks; no model or credentials are used."""
import copy
import math
import unittest
import evaluate_output_probe_qwen35_27b as E
import adjective_common_qwen35_27b as C
from adv_l1_behavioral_qwen35_27b import compare


def obj(label,x,y,w=1.,h=1.,anchor=None,description='',rotation=0.):
    return dict(label=label,bbox=[x,y,w,h],position=[x+w/2,y+h/2],rotation_angle=rotation,
        anchor=anchor,description=description,concise_description='',object_type='bulk')


def run(objects,assets=None):
    room={'room_dimension_x':10,'room_dimension_y':10}
    return dict(scene={'objects':objects},benchmark_copy={'room_dimension':room,'assets':assets or {'bed':1}},
        labels=C.scene_labels(objects,room),calls=[dict(stage=stage,output_text='same') for stage in C.STAGES])


class EvaluationTests(unittest.TestCase):
    def test_exact_union_overlap_and_touch(self):
        self.assertAlmostEqual(E.union_area([(0,0,2,2),(1,0,3,2)]),6)
        self.assertAlmostEqual(E.union_area([(0,0,1,1),(1,0,2,1)]),2)
        self.assertEqual(E.union_area([]),0)

    def test_room_clip_and_clearance(self):
        a=run([obj('bed',4,0,2,2),obj('chair',0,0)],{'bed':1,'chair':1})
        self.assertAlmostEqual(E.features(a)['free_space'],.97)
        self.assertAlmostEqual(E.features(a)['clearance'],3/math.sqrt(200))
        self.assertEqual(set(E.FEATURES),{'free_space','clearance','regularity','spacing_consistency'})

    def test_zero_gap_consistency_not_order(self):
        self.assertIsNone(E.consistency([0.,0.]))
        self.assertAlmostEqual(E.consistency([1.,1.,1.]),1)

    def test_identity_and_duplicate_object_matching(self):
        a=run([obj('bed',0,0),obj('bed',2,0)],{'bed':2})
        b=run([obj('bed',1,0),obj('bed',3,0)],{'bed':2})
        profile=E.object_profile(a,b)
        self.assertEqual([r['occurrence'] for r in profile],[1,2])
        self.assertEqual([r['position_shift'] for r in profile],[1,1])
        q=E.specificity(a,b,'spacious',profile)
        self.assertEqual(q['object_set_jaccard'],1)
        self.assertEqual(q['required_inventory_recall'],1)
        self.assertEqual(q['collateral_change_rate'],0)
        self.assertEqual(q['semantic_structural_ratio'],1)

    def test_rotation_wrap(self):
        a=run([obj('bed',0,0,rotation=350)])
        b=run([obj('bed',0,0,rotation=10)])
        self.assertEqual(E.object_profile(a,b)[0]['rotation_change_degrees'],20)

    def test_no_brightness_or_quiet_semantics(self):
        a=run([obj('bed',0,0),obj('chair',2,0)],{'bed':1,'chair':1})
        for condition in ('bright','quiet'):
            s=E.semantic(a,a,condition)
            self.assertEqual(s['semantic_status'],E.UNOBSERVABLE)
            self.assertIsNone(s['direction_accuracy'])
            self.assertIsNone(s['semantic_direction_cosine'])
            q=E.specificity(a,a,condition,E.object_profile(a,a))
            self.assertIsNone(q['collateral_change_rate'])
            self.assertIsNone(q['semantic_structural_ratio'])

    def test_direction_zero_is_not_improvement(self):
        a=run([obj('bed',0,0),obj('chair',3,0)],{'bed':1,'chair':1})
        s=E.semantic(a,a,'spacious')
        self.assertEqual(s['direction_accuracy'],0)
        self.assertIsNone(s['semantic_direction_cosine'])
        self.assertEqual(s['normalized_signed_effect']['free_space'],0)

    def test_relation_consistency_and_vertical_unsupported(self):
        a=run([obj('bed',0,0),obj('chair',2,0,anchor='bed',description='A chair to the left of the bed.'),obj('shelf',0,2,anchor='bed',description='A shelf above the bed.')],{'bed':1,'chair':1,'shelf':1})
        scores,rows=E.relations(a)
        self.assertEqual(scores['evaluated_count'],1)
        self.assertEqual(scores['src'],0)
        self.assertEqual(scores['unevaluated_count'],1)
        self.assertFalse(next(r for r in rows if r['relation']=='vertical')['evaluatable'])
        a['scene']['objects'][1]['description']='A chair to the right of the bed.'
        self.assertEqual(E.relations(a)[0]['src'],1)

    def test_ambiguous_anchor_not_guessed(self):
        a=run([obj('bed',0,0),obj('bed',2,0),obj('chair',4,0,anchor='bed',description='A chair near the bed.')],{'bed':2,'chair':1})
        self.assertIsNone(E.relations(a)[0]['src'])
        v,checks,_=E.validity(a)
        self.assertTrue(any(c['violated'] for c in checks if c['check']=='anchor_unambiguous'))

    def test_constraint_checks_no_mutation(self):
        a=run([obj('bed',0,0),obj('chair',.5,0),obj('shelf',5,0)],{'bed':1,'chair':1,'shelf':1,'desk':1})
        before=copy.deepcopy(a)
        v,checks,_=E.validity(a)
        self.assertEqual(a,before)
        self.assertEqual(v['missing_required_objects'],1)
        self.assertEqual(v['oob_objects'],1)
        self.assertEqual(v['overlap_pairs'],1)
        self.assertGreater(v['constraint_violation_rate'],0)

    def test_l1_same_definition(self):
        a=run([obj('bed',0,0)])
        b=run([obj('bed',1,0)])
        values=compare(a,b)
        self.assertEqual(values['object_set_diff'],0)
        self.assertEqual(values['centroid_shift'],1)
        self.assertFalse(values['structural_divergence'])

    def test_single_benchmark_l2_statistic_is_undefined(self):
        matrix=[[None,.4],[.4,None]]
        self.assertIsNone(C.distance_statistic(matrix,[0,1]))
        result=C.permutation_test(matrix,1,paired=True)
        self.assertIsNone(result['test_statistic'])
        self.assertIsNone(result['p_value'])
        self.assertEqual(result['number_of_permutations'],0)


if __name__=='__main__':
    unittest.main()
