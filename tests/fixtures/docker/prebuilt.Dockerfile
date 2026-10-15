# Mimics a SWE-bench instance image: repo baked in at /testbed at its base commit, plus
# environment leftovers that the sandbox must preserve (ignored build artifact) or hide
# from diffs (untracked file not covered by .gitignore). Build context: tests/fixtures.
FROM ablation-fixture:py3.12
COPY calc_repo /testbed
WORKDIR /testbed
RUN printf '*.so\n' > .gitignore \
    && git init -q && git add -A \
    && git -c user.email=fixture@localhost -c user.name=fixture commit -q -m base \
    && printf 'binary' > calc/_speedups.so \
    && printf 'left by the environment build\n' > shipped_untracked.txt
