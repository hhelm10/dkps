"""Information boundaries for the additional score, temporal, and style probes."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'projects/pkps/scripts'))
from experiments.stress_next_checks import additive_predict, additive_fold, temporal_design
from experiments.stress_completion import inner_split
from data.embed_stress_extensions import style_text, restore_text
from data.prepare_eee_followups import selected_runs


@pytest.mark.parametrize('scale',['raw','logit-standardized'])
def test_additive_target_task_does_not_estimate_its_own_strength(scale):
    reference=np.array([[.2,.4,.5],[.4,.5,.7],[.3,.6,.8]])
    target=np.array([[.2,.3,.4]])
    before=additive_predict(reference,target,scale)
    changed=target.copy(); changed[0,1]=.99
    after=additive_predict(reference,changed,scale)
    assert before[0,1]==pytest.approx(after[0,1])
    assert not np.allclose(before[:,[0,2]],after[:,[0,2]])


def test_additive_recovers_balanced_additive_structure():
    task=np.array([.3,.4,.5,.6]); strength=np.array([-.1,0,.1])
    ref=task+strength[:,None]
    sample=task[None,:]+.05
    np.testing.assert_allclose(additive_predict(ref,sample),sample,atol=1e-14)


def test_additive_full_family_and_cache_only_boundaries():
    models=['a/1','a/2','b/1','b/2','c/1','c/2']
    sample=np.random.default_rng(8).uniform(.1,.9,(6,3))
    with pytest.raises(ValueError,match='outer family'):
        additive_fold(models,sample,sample.copy(),'a','full-reference')
    with pytest.raises(ValueError,match='cache-only'):
        additive_fold(models,sample,sample.copy(),'a','cache-only')
    permitted=sample.copy(); permitted[:2]=np.nan
    _,pred,_=additive_fold(models,sample,permitted,'a','full-reference')
    changed=sample.copy(); changed[0,0]=.999
    _,after,_=additive_fold(models,changed,permitted,'a','full-reference')
    for key in pred: assert pred[key][0,0]==pytest.approx(after[key][0,0])


def test_temporal_split_uses_evaluation_time_and_reports_cold_start():
    truth=pd.DataFrame([dict(model_id=m,task_id=t,suite=t,y_full=.5)
                        for m in ['a/1','b/1','c/1'] for t in ['x','y']])
    times=['2026-01-01','2026-01-02','2026-01-03','2026-03-01','2026-04-01','2026-04-02']
    metadata=[dict(model_id=r.model_id,suite=r.suite,evaluation_timestamp=d,added_at='2026-12-01')
              for r,d in zip(truth.itertuples(),times)]
    early,test,models,tasks,counts=temporal_design(truth,metadata,.5)
    assert set(models)=={'a/1','b/1'}
    assert len(early)==3 and len(test)==1 and counts['n_cold_start_cells']==2
    assert set(zip(early.model_id,early.task_id)).isdisjoint(zip(test.model_id,test.task_id))
    changed=truth.assign(y_full=999)
    assert temporal_design(changed,metadata,.5)[-1]==counts
    metadata[0]['evaluation_timestamp']=None
    with pytest.raises(ValueError,match='timestamps'): temporal_design(truth,metadata,.5)


def test_natural_holdout_never_invents_ground_truth():
    natural=np.array([[1,1,0],[1,0,1],[0,1,1]],dtype=bool)
    train,validation=inner_split(natural,0,.3)
    assert not (validation & ~natural).any()
    assert np.array_equal(train|validation,natural)
    assert train.any(0).all() and train.any(1).all()


@pytest.mark.parametrize('variant',['original','json','blockquote'])
def test_style_interventions_exactly_preserve_response_content(variant):
    text='Answer: α = 2.\n\n```python\nif x:\n    print("yes")\n```\n'
    assert restore_text(style_text(text,variant),variant)==text


def test_all_model_selection_preserves_native_missingness():
    manifest={'files':{f'data/{b}/{m}/run_samples.jsonl':{'added_at':'2026-01-01'}
                       for b,m in [('x','a/1'),('x','b/1'),('y','a/1')]}}
    shared=selected_runs(manifest,benchmarks=['x','y'])
    sparse=selected_runs(manifest,benchmarks=['x','y'],shared_only=False)
    assert len(shared)==2 and len(sparse)==3 and ('y','b/1') not in sparse
