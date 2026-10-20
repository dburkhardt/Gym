"""Pinned MOOSE-Chem orchestration, preserving creator stages and prompts."""
import argparse
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

MODEL = 'nvidia/nvidia/nemotron-3.5-super-vl'
PIN = 'f8cda98a3bed30b0f0d85fba625e343d688461ab'


def creator_defaults(path):
    parser = argparse.ArgumentParser()
    calls = [n for n in ast.walk(ast.parse(Path(path).read_text()))
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Attribute)
             and isinstance(n.value.func.value, ast.Name)
             and n.value.func.value.id == 'parser' and n.value.func.attr == 'add_argument']
    exec(compile(ast.Module(body=calls, type_ignores=[]), str(path), 'exec'),
         {'parser': parser, 'str': str, 'int': int, 'float': float})
    return parser.parse_args([])


class ReplayPrefix:
    def __init__(self, path):
        self.rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
        self.index = 0
        last = self.rows[-1]['response']['choices'][0]
        self.failed_tail = last['finish_reason'] == 'length' and last['message']['content'] is None
        assert self.failed_tail or all(r['response']['choices'][0]['message']['content'] is not None for r in self.rows)

    def consume(self, stage, request):
        if self.index >= len(self.rows):
            return None
        row = self.rows[self.index]
        comparable = dict(request)
        assert row['request']['max_tokens'] in (8192,16384)
        comparable['max_tokens'] = row['request']['max_tokens']
        assert row['stage'] == stage and row['request'] == comparable, f'Replay divergence at {self.index}'
        self.index += 1
        if self.index == len(self.rows) and self.failed_tail:
            return None  # failed request is retried only after its exact identity is checked
        from openai.types.chat import ChatCompletion
        return ChatCompletion.model_validate(row['response'])


class RecordedClient:
    def __init__(self, *, stage, trace, replay=None, **kwargs):
        from openai import OpenAI
        assert kwargs['base_url'] == 'http://127.0.0.1:49180/v1'
        self.inner = OpenAI(**kwargs, max_retries=0, timeout=600)
        self.stage, self.trace, self.replay = stage, Path(trace), replay
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        assert kwargs['model'] == MODEL
        assert kwargs.get('max_tokens') == 8192  # creator default, never silently changed
        if self.replay is not None:
            result = self.replay.consume(self.stage, kwargs)
            if result is not None:
                with self.trace.with_name('replay-receipts.jsonl').open('a') as stream:
                    stream.write(json.dumps({'stage':self.stage,'source_index':self.replay.index-1,'inference':False})+'\n')
                return result
            kwargs['max_tokens'] = 16384
        started = time.time()
        record = {'stage': self.stage, 'request': kwargs, 'started': started}
        try:
            result = self.inner.chat.completions.create(**kwargs)
            assert result.model == MODEL
            record['response'] = result.model_dump()
            return result
        except Exception as exc:
            record['error_type'] = type(exc).__name__
            raise
        finally:
            record['elapsed_seconds'] = time.time() - started
            with self.trace.open('a') as stream:
                stream.write(json.dumps(record) + '\n')


def run(repo, out, task_id, *, screening_only=False, replay_path=None):
    repo, out = Path(repo).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pins = json.loads(Path(__file__).with_name('source-sha256.json').read_text())
    for relative, expected in pins.items():
        assert hashlib.sha256((repo/relative).read_bytes()).hexdigest() == expected, relative
    sys.path.insert(0, str(repo))
    names = ['inspiration_screening', 'hypothesis_generation', 'evaluate']
    modules = [importlib.import_module('Method.' + name) for name in names]
    trace = out / 'model-traces.jsonl'
    if trace.exists():
        raise RuntimeError('Existing model trace; reconcile explicitly before rerunning')
    replay = ReplayPrefix(replay_path) if replay_path else None
    for stage, module in zip(names, modules):
        module.OpenAI = lambda _stage=stage, **kw: RecordedClient(stage=_stage, trace=trace, replay=replay, **kw)
    common = dict(api_type=0, api_key=Path(os.environ['MOOSE_KEY_FILE']).read_text().strip(),
                  base_url='http://127.0.0.1:49180/v1', model_name=MODEL,
                  chem_annotation_path=str(repo/'Data/chem_research_2024.xlsx'),
                  custom_inspiration_corpus_path=str(repo/'Data/Inspiration_Corpus_150.json'),
                  custom_research_background_path='', corpus_size=150,
                  background_question_id=task_id, if_save=1,
                  if_use_strict_survey_question=1, if_use_background_survey=1)
    args = [creator_defaults(repo/'Method'/(name+'.py')) for name in names]
    for arg in args:
        vars(arg).update(common)
    vars(args[0]).update(num_screening_window_size=15, num_screening_keep_size=3,
                         num_round_of_screening=4, if_select_based_on_similarity=0,
                         output_dir=str(out/'screening.json'))
    screening = modules[0].Screening(args[0]); screening.run()
    if screening_only:
        return {'scope':'creator_inspiration_retrieval_only', 'screening':json.loads((out/'screening.json').read_text())}
    vars(args[1]).update(inspiration_dir=str(out/'screening.json'),output_dir=str(out/'hypotheses.json'),
                         if_load_from_saved=0,if_use_gdth_insp=0,idx_round_of_first_step_insp_screening=2,
                         num_mutations=3,num_itr_self_refine=3,num_self_explore_steps_each_line=3,
                         num_screening_window_size=12,num_screening_keep_size=3,
                         if_mutate_inside_same_bkg_insp=1,if_mutate_between_diff_insp=1,if_self_explore=0,
                         if_consider_external_knowledge_feedback_during_second_refinement=0,
                         inspiration_ids=[-1],recom_inspiration_ids=[],recom_num_beam_size=5,
                         self_explore_inspiration_ids=[],self_explore_num_beam_size=5,max_inspiration_search_steps=3)
    generation = modules[1].HypothesisGenerationEA(args[1])
    modules[1].hyp_gene_ea = generation  # creator __main__ binds this global used by later stages
    generation.hypothesis_generation_for_one_background_question(background_question_id=task_id,
                             inspiration_ids=args[1].inspiration_ids,final_data_collection={})
    vars(args[2]).update(hypothesis_dir=str(out/'hypotheses.json'),output_dir=str(out/'evaluation.json'),
                         if_load_from_saved=0,if_with_gdth_hyp_annotation=1)
    evaluator = modules[2].Evaluate(args[2]); evaluator.run()
    return {'scope':'creator_full_pipeline_150_corpus_Nemotron_all_roles_variant',
            'evaluation':json.loads((out/'evaluation.json').read_text()),
            'replay_source':str(replay_path) if replay_path else None,
            'reused_successful_responses':sum(1 for _ in (out/'replay-receipts.jsonl').open()) if (out/'replay-receipts.jsonl').exists() else 0,
            'budget_protocol':'saved_8192_and_16384_prefix_then_16384_new_calls' if replay_path else 'creator_8192'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task-id',type=int,required=True)
    parser.add_argument('--screening-only',action='store_true')
    parser.add_argument('--replay-path',type=Path)
    args=parser.parse_args()
    try:
        result=run(args.repo,args.output,args.task_id,screening_only=args.screening_only,replay_path=args.replay_path)
        (args.output/'result.json').write_text(json.dumps(result,indent=2))
    except Exception as exc:
        (args.output/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc)},indent=2))
        raise
