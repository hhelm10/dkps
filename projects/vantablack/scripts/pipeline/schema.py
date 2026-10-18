"""Canonical tables for the vantablack pipeline.

Every data source (synthetic simulator, archived crawl, fresh crawl) is
converted into these tables before anything downstream runs. Times are
float hours since an origin recorded in `meta['origin']`.

actions   one row per observed action
    action_id    str    unique
    agent_id     str    actor
    action_type  str    post | comment | upvote | downvote
    t            float  hours since origin
    submolt      str
    thread_id    str    root post of the thread the action lives in
    parent_id    str    post/comment replied to or voted on ('' for posts)
    target_agent str    author of parent_id ('' if unknown / post)
    text         str    '' for votes
    score_at_fetch float  net votes on this action at fetch time (NaN if n/a)

contexts  one row per thread
    thread_id, submolt, t, text

score_events  one row per score increment received by an agent
    agent_id (receiver), t, value   (vote-level if attributable; else
    differences between repeated snapshots)
"""
import numpy as np
import pandas as pd

ACTION_COLS = ['action_id', 'agent_id', 'action_type', 't', 'submolt',
               'thread_id', 'parent_id', 'target_agent', 'text',
               'score_at_fetch']
CONTEXT_COLS = ['thread_id', 'submolt', 't', 'text']
SCORE_COLS = ['agent_id', 't', 'value']
CONTENT_TYPES = ('post', 'comment')


def validate(actions, contexts, score_events):
    for df, cols, name in [(actions, ACTION_COLS, 'actions'),
                           (contexts, CONTEXT_COLS, 'contexts'),
                           (score_events, SCORE_COLS, 'score_events')]:
        missing = set(cols) - set(df.columns)
        assert not missing, f'{name} missing {missing}'
    assert actions.action_id.is_unique
    assert contexts.thread_id.is_unique
    content = actions[actions.action_type.isin(CONTENT_TYPES)]
    unknown = set(content.thread_id) - set(contexts.thread_id)
    assert not unknown, f'{len(unknown)} threads lack a context row'


def coverage(actions):
    """Coverage numbers the TODO list asks us to record."""
    a = actions
    content = a.action_type.isin(CONTENT_TYPES)
    votes = a.action_type.isin(('upvote', 'downvote'))
    return {
        'n_actions': int(len(a)),
        'n_agents': int(a.agent_id.nunique()),
        'n_threads': int(a.thread_id.nunique()),
        'frac_content_with_text': float((a.text[content].str.len() > 0).mean())
        if content.any() else np.nan,
        'n_vote_records': int(votes.sum()),
        'frac_votes_with_voter': float((a.agent_id[votes] != '').mean())
        if votes.any() else np.nan,
        'by_type': a.action_type.value_counts().to_dict(),
    }
