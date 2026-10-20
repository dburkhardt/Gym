import os
from pathlib import Path
import sys
import pytest
from benchmarks.moosechem.native import creator_defaults

REPO=Path(os.environ['MOOSE_SOURCE']).resolve()
sys.path.insert(0,str(REPO))
from Method.inspiration_screening import Screening
from Method import inspiration_screening

@pytest.fixture
def screen():
    args=creator_defaults(REPO/'Method/inspiration_screening.py')
    vars(args).update(api_type=0,api_key='unused-control',base_url='http://127.0.0.1:49180/v1',
        chem_annotation_path=str(REPO/'Data/chem_research_2024.xlsx'),
        custom_inspiration_corpus_path=str(REPO/'Data/Inspiration_Corpus_150.json'),
        num_screening_window_size=15,num_screening_keep_size=3,if_use_strict_survey_question=1)
    return Screening(args)

def test_reference_recall_controls(screen):
    question=screen.bkg_q_list[18]
    gold=screen.dict_bkg2insp[question]
    assert screen.check_how_many_hit_groundtruth_insp(question,[[[title,None]] for title in gold])==[1,1]
    assert screen.check_how_many_hit_groundtruth_insp(question,[])==[0,0]

def test_retrieval_prompt_reference_isolation(screen,monkeypatch):
    captured=[];candidates=screen.title_abstract_collector[:15]
    def fake(prompt,*args,**kwargs):
        captured.append(prompt)
        return [{'Title':x[0]} for x in candidates[:3]]
    monkeypatch.setattr(inspiration_screening,'llm_generation_with_multiple_extractions',fake)
    question=screen.bkg_q_list[18]
    results,next_candidates=screen.one_round_screening(question,candidates)
    assert len(results)==1 and len(next_candidates)==3
    assert question in captured[0] and screen.dict_bkg2survey[question] in captured[0]
    assert screen.dict_bkg2groundtruthHyp[question] not in captured[0]
    assert screen.dict_bkg2note[question] not in captured[0]

def test_defaults_are_source_derived():
    assert creator_defaults(REPO/'Method/hypothesis_generation.py').if_use_gdth_insp==0
    assert creator_defaults(REPO/'Method/evaluate.py').api_type==1
