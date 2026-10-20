# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import unittest
from native_score import annotate
class Controls(unittest.TestCase):
    def test_native_zero_is_not_failed(self):
        r=annotate({'total_score':0},[{'parsed':{'score':0}}]);self.assertEqual(r['reward'],0);self.assertFalse(r['mask_sample'])
    def test_invalid_judge_does_not_become_performance_zero(self):
        for result in (None,{}, {'reasoning':'no score'}, {'score':True}, {'score':'bad'}):
            r=annotate({'total_score':0},[{'parsed':result}]);self.assertIsNone(r['reward']);self.assertTrue(r['mask_sample']);self.assertEqual(r['native']['total_score'],0)
    def test_native_components_preserved(self):
        r=annotate({'total_score':51.25,'items':[{'score':50},{'score':55}]},[{'parsed':{'score':50}},{'parsed':{'score':55}}]);self.assertEqual(r['reward'],.5125);self.assertEqual(len(r['native']['items']),2)
    def test_missing_report_is_explicit(self):
        self.assertTrue(annotate({'error':'No report found'},[])['mask_sample'])
if __name__=='__main__':unittest.main()
