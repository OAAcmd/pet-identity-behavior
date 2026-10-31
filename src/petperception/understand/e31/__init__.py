"""The E31 behaviour-explanation system (v1), frozen: verbatim copies of vocab, knowledge, question, recognize and explain at
commit 0c1f7c5 (only the features import is re-pointed to the shared extractor, whose version-1 output is unchanged).

E32's pre-registered decision (``results/e32/metrics.json`` "decision") keeps this system as the default of
``petperception explain`` because v2 missed an absolute target, so it must stay exactly as tested: do not edit these files.
``results/e32/default_v1_check.json`` records that ``Explainer.load(version=1)`` reproduces the tested v1 answers.
"""
