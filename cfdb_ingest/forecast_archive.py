"""
The archive side of a forecast pipeline: opening a ``grid_forecast`` target (an S3-backed EDataset,
or a plain local file when no remote is configured), refusing to create one by accident, retention as
a pure function plus its application, and a push that never leaves the remote advertising an init
whose chunks did not land -- plus the process guards (a run flock, a SIGTERM handler, a pending
sidecar) that make an interrupted run recoverable.

Lifted from ``ifs-download`` 0.1 (``ifs_dl/archive.py`` and ``run.py``, reviewed dual-blind in
rounds ``ifs-download-plan-1`` / ``ifs-download-code-1``) so that every pipeline writing a forecast
archive -- the IFS downloader, the WRF forecast ingest -- shares one protocol. What is NOT lifted,
deliberately: the IFS-specific completeness gates around ``IfsIngest.convert`` (they assume a whole
init per call) and the "discard the working file of an interrupted ingest" recovery step (it assumes
nothing reached the remote, which is false once an init is pushed incrementally; an incremental
writer reconciles with :func:`forecast.missing_chunks` instead).

The ebooklet facts this leans on (verified against 0.10.5): a fresh local file opened with
``flag='c'`` attaches to an existing remote; ``truncate`` through the handle journals deletes of
chunks that were never pulled; a partial push COMMITS the metadata and keeps the failed keys only in
the local journal, so the local file must survive a failed push; the write lock is released on
``close()`` and has no expiry -- ``force_lock=True`` breaks only tickets older than two hours, so a
ticket left by a process that died minutes ago needs :func:`break_locks`.
"""

import contextlib
import datetime
import fcntl
import json
import logging
import os
import pathlib
import signal
from collections.abc import Iterable, Sequence
from typing import Optional, Union

import numpy as np
import urllib3
from cfdb import open_dataset, open_edataset
from ebooklet import S3Connection
from ebooklet.errors import RemoteMissingError

from cfdb_ingest import forecast as fc

log = logging.getLogger(__name__)

DATASET_TYPE = 'grid_forecast'
SIDECAR_NAME = 'pending.json'
LOCK_NAME = '.lock'


class WouldCreate(Exception):
    """The target has no dataset (locally or remotely) and creating one was not allowed."""


class PushFailed(Exception):
    """Chunks failed to upload after the forced retry; the local file holds the journal."""


# ---------------------------------------------------------------------- opening


def open_handle(
    remote: Union[dict, str, None],
    path: pathlib.Path,
    flag: str,
    *,
    dataset_type: str,
    lock_timeout: int,
    force_lock: bool,
):
    """The raw handle (caller closes), or None when ``flag='r'`` finds no dataset (locally or on the remote)."""
    if remote is None:
        if flag == 'r' and not path.exists():
            return None
        return open_dataset(str(path), flag, dataset_type=dataset_type)
    try:
        return open_edataset(
            remote,
            str(path),
            flag=flag,
            dataset_type=dataset_type,
            num_groups=None,
            lock_timeout=lock_timeout,
            force_lock=force_lock,
        )
    except RemoteMissingError:
        if flag != 'r':
            raise
        return None
    except urllib3.exceptions.HTTPError as e:
        # ebooklet surfaces a 403 on the db object as an unparseable-XML error (a HEAD has no body);
        # the cause is nearly always an application key without readFiles/writeFiles/deleteFiles
        if 'ParseError' in str(e) or 'AccessDenied' in str(e) or 'not entitled' in str(e):
            bucket = remote.get('bucket') if isinstance(remote, dict) else remote
            key = remote.get('db_key') if isinstance(remote, dict) else ''
            raise PermissionError(
                f'the S3 key is not entitled to {bucket}/{key} ({e}); '
                f'ebooklet needs readFiles, writeFiles, deleteFiles and listFiles on that bucket'
            ) from e
        raise


@contextlib.contextmanager
def open_target(
    remote: Union[dict, str, None],
    path,
    flag: str,
    *,
    dataset_type: str = DATASET_TYPE,
    lock_timeout: int = 300,
    force_lock: bool = False,
):
    """
    Yield the dataset handle: ``open_edataset`` when a remote connection (a ``S3Connection`` kwargs
    dict, or a public ``db_url`` string for ``flag='r'``) is given -- ``path`` is then the local
    working file -- else a plain ``open_dataset``. ``flag='r'`` on a target that does not exist yields
    None; every other failure (credentials, network, a corrupt file) propagates. Always closed on exit;
    for an EDataset that is what releases the write lock.
    """
    ds = open_handle(remote, pathlib.Path(path), flag, dataset_type=dataset_type, lock_timeout=lock_timeout,
                     force_lock=force_lock)
    try:
        yield ds
    finally:
        if ds is not None:
            ds.close()


def break_locks(remote: dict, *, older_than: Optional[datetime.datetime] = None) -> list:
    """
    Remove other writers' lock tickets on the remote db object. Default: EVERY ticket (``older_than``
    = now) -- for a writer that knows it is the only one by construction (a job-level flock plus one
    job per archive at a time) and is recovering from a process of its own that died holding the
    lock. ``force_lock=True`` on open is not enough for that case: it is age-gated to two hours.
    Returns what was removed.
    """
    ts = older_than if older_than is not None else datetime.datetime.now(datetime.timezone.utc)
    with S3Connection(**remote).open('w') as session:
        return session.break_other_locks(timestamp=ts)


def refuse_create(ds, allow_create: bool) -> bool:
    """True when the dataset is empty (i.e. this run would create it); raises unless allowed."""
    empty = len(ds.coord_names) == 0
    if empty and not allow_create:
        raise WouldCreate(
            'the target has no dataset: a scheduled run never creates one (a wrong db_key or a transient '
            '404 on the db object would otherwise create a one-init dataset that overwrites the archive '
            'on push). Allow creation explicitly for the first run.'
        )
    return empty


def done_inits(ds) -> list:
    """``complete_inits`` as ISO strings (``YYYY-MM-DDTHH:MM``), sorted."""
    return sorted(fc.complete_inits(ds)) if ds is not None else []


# ---------------------------------------------------------------------- retention


def inits_to_drop(
    complete: Iterable, keep_days: Optional[int], protect: Iterable = (), min_keep: int = 2
) -> tuple:
    """
    ``(cutoff, dropped)`` for ``forecast_reference_time.truncate(start=cutoff)`` -- a PREFIX cut, so
    the cutoff is a single value: the oldest complete init that survives.

    Works on the COMPLETE inits only (auto-filled empty slots are not inits). ``cutoff`` = the oldest
    complete init at or after ``max(complete) - keep_days``, lowered to ``min(protect)`` so a
    back-filled old init written this run is never cut, and raised no further than leaves
    ``min_keep`` complete inits. Disabled (``(None, [])``) when ``keep_days`` is falsy.

    Note that ``truncate`` is positional: an UNMARKED init (a run that never finished) older than the
    cutoff is dropped with the marked ones, and one younger survives until the cutoff passes it.
    """
    if not keep_days or keep_days <= 0:
        return None, []
    if min_keep < 1:
        raise ValueError('min_keep must be at least 1 (truncating every init is refused by cfdb anyway)')
    inits = sorted({np.datetime64(i, 'm') for i in complete})
    if not inits:
        return None, []
    horizon = inits[-1] - np.timedelta64(int(keep_days) * 24 * 60, 'm')
    cutoff = next(i for i in inits if i >= horizon)  # inits[-1] always qualifies
    protected = [np.datetime64(p, 'm') for p in protect]
    if protected:
        cutoff = min(cutoff, min(protected))
    dropped = [i for i in inits if i < cutoff]
    keep_n = max(min_keep, len(inits) - len(dropped))
    dropped = inits[: max(0, len(inits) - keep_n)]
    if not dropped:
        return None, []
    return inits[len(dropped)], dropped


def apply_retention(
    ds,
    *,
    keep_days: Optional[int],
    protect: Iterable = (),
    min_keep: int = 2,
    dry_run: bool = False,
    label: str = 'retention',
) -> tuple:
    """
    Truncate ``forecast_reference_time`` at the cutoff from ``inits_to_drop`` and remove the dropped
    inits from ``complete_inits`` (``truncate`` does not touch attrs). ``dry_run`` computes only.
    ``label`` names the pipeline in the history line.
    """
    cutoff, dropped = inits_to_drop(done_inits(ds), keep_days, protect, min_keep)
    dropped_s = [str(d) for d in dropped]
    if cutoff is None or dry_run:
        return cutoff, dropped_s
    ds[fc.FRT].truncate(start=cutoff)
    for init in dropped:
        fc.unmark_init_complete(ds, init)
    fc.append_history(
        ds,
        f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {label}: dropped {dropped_s}, cutoff {cutoff}',
    )
    return cutoff, dropped_s


# ---------------------------------------------------------------------- push


def remove_local(path) -> None:
    """Delete a local cfdb working file together with ebooklet's ``<name>.remote_index`` sidecar."""
    path = pathlib.Path(path)
    path.unlink(missing_ok=True)
    path.with_name(path.name + '.remote_index').unlink(missing_ok=True)


def push_twice(ds) -> None:
    """``push()``, one forced retry, ``PushFailed`` if chunks still failed."""
    result = ds.push()
    if result.failures:
        log.warning('push: %d failures, retrying with force_push', len(result.failures))
        result = ds.push(force_push=True)
    if result.failures:
        raise PushFailed(f'{len(result.failures)} chunk uploads failed after retry: {list(result.failures)[:3]}')


def push_and_mark(ds, inits: Iterable) -> None:
    """
    Publish in two commits so the remote never advertises an init whose chunks have not landed:
    un-mark ``inits`` -> push the chunks (one forced retry) -> mark them complete -> push the metadata.
    ebooklet commits metadata on ANY partially successful push, so the marks must not ride the chunk
    push. On failure the init stays un-marked on the remote; the caller keeps the local file (whose
    journal holds the failed keys) and the next run finishes the job through ``recover``.

    An incremental writer pushes its partial calls with a plain ``push_twice`` (the init is unmarked
    throughout) and calls this once, at the end, after ``forecast.missing_chunks`` is empty.
    """
    inits = list(inits)
    for init in inits:
        fc.unmark_init_complete(ds, init)
    push_twice(ds)
    if inits:
        for init in inits:
            fc.mark_init_complete(ds, init)
        push_twice(ds)


def recover(ds, inits: Sequence) -> None:
    """Finish a push that was interrupted last run: the same two commits as ``push_and_mark``."""
    push_and_mark(ds, inits)


# ---------------------------------------------------------------------- process guards


def sidecar_path(work_dir) -> pathlib.Path:
    return pathlib.Path(work_dir) / SIDECAR_NAME


def write_sidecar(work_dir, stage: str, inits: Sequence, **extra) -> None:
    """Record what the run is in the middle of (``stage``: e.g. 'ingest' | 'push') for the next run."""
    pathlib.Path(work_dir).mkdir(parents=True, exist_ok=True)
    sidecar_path(work_dir).write_text(json.dumps({'stage': stage, 'inits': list(inits), **extra}))


def read_sidecar(work_dir) -> Optional[dict]:
    p = sidecar_path(work_dir)
    return json.loads(p.read_text()) if p.exists() else None


def clear_sidecar(work_dir) -> None:
    sidecar_path(work_dir).unlink(missing_ok=True)


@contextlib.contextmanager
def run_lock(work_dir, *, holder: str = 'another run'):
    """flock on ``<work_dir>/.lock``: the overlap guard for one pipeline's writes to one archive."""
    work_dir = pathlib.Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(work_dir / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise RuntimeError(f'{holder} holds {work_dir / LOCK_NAME}') from e
        yield
    finally:
        os.close(fd)


def install_sigterm_handler() -> None:
    """
    ``docker stop`` / SLURM / a parent's ``terminate()`` send SIGTERM, whose default action skips
    ``finally`` blocks and so would leave the remote write lock's ticket behind. Turn it into
    SystemExit so every ``with`` closes its dataset.
    """

    def _handler(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handler)
