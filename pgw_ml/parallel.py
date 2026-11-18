"""
Robust parallel map.

multiprocessing.Pool replaces a worker that dies (for example, killed by a
memory limit) but loses its task, and imap then waits forever. This helper
uses concurrent.futures.ProcessPoolExecutor, which detects a dead worker
(BrokenProcessPool). The executor is then recreated and only the unfinished
tasks are resubmitted, up to 'retries' times. Results are yielded as they
complete (unordered).
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool


def robust_map(func, jobs, nproc, initializer=None, initargs=(), retries=2, log=print):
    pending = list(enumerate(jobs))
    attempt = 0
    while pending:
        done = set()
        try:
            with ProcessPoolExecutor(max_workers=max(1, nproc), initializer=initializer,
                                     initargs=initargs) as ex:
                futs = {ex.submit(func, job): i for i, job in pending}
                for fut in as_completed(futs):
                    res = fut.result()          # raises BrokenProcessPool if a worker died
                    done.add(futs[fut])
                    yield res
            pending = []
        except BrokenProcessPool:
            pending = [(i, job) for i, job in pending if i not in done]
            attempt += 1
            if attempt > retries:
                raise RuntimeError(f"{len(pending)} tasks lost after {retries} retries: a worker "
                                   "process died (memory limit?). Reduce NPROC or CHUNK.")
            log(f"  [WARNING] a worker process died; retrying {len(pending)} unfinished tasks "
                f"(attempt {attempt}/{retries})")
