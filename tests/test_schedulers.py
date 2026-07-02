# Copyright (c) 2026 Hewlett Packard Enterprise Development LP
# MIT License
"""Regression tests for how ``ModelEntry`` initialises ``last_run_at``.

These tests involve NO broker, NO queue and NO worker: the only code under
test is ``ModelEntry`` (rdbbeat/schedulers.py) plus ``TzAwareCrontab.is_due``
(rdbbeat/tzcrontab.py).  Whatever they show cannot be explained by a queue
overflowing, because the skip decision happens *before* anything could ever
be enqueued.

Scenario used throughout: a task with the daily crontab ``0 10 * * *`` (UTC)
that beat loads from the database 30 seconds AFTER today's 10:00 occurrence
(e.g. beat was (re)started at 10:00:30, or the row was committed moments
before 10:00 and picked up on the next schedule reload).
"""

import datetime as dt
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import pytz
from celery import Celery

from rdbbeat.db.models import CrontabSchedule, PeriodicTask
from rdbbeat.schedulers import ModelEntry
from rdbbeat.tzcrontab import TzAwareCrontab

# Today's occurrence of the ``0 10 * * *`` (UTC) crontab.
SCHEDULED_TIME = dt.datetime(2026, 7, 1, 10, 0, 0, tzinfo=pytz.utc)
# Beat loads the entry 30 seconds after the occurrence has passed.
LOAD_TIME = SCHEDULED_TIME + dt.timedelta(seconds=30)
ONE_DAY = 24 * 60 * 60


@pytest.fixture
def app():  # noqa: ANN201
    return Celery(set_as_current=False)


@contextmanager
def loaded_entry(app, last_run_at=None):  # noqa: ANN001, ANN201
    """Simulate beat loading a ``PeriodicTask`` row from the DB at ``LOAD_TIME``.

    The wall clock is frozen at ``LOAD_TIME`` for both time sources involved:
    ``app.now()`` (used by ``ModelEntry._default_now``) and
    ``TzAwareCrontab.nowfunc`` (used by ``is_due``/``remaining_estimate``).
    """
    crontab = CrontabSchedule(
        minute="0",
        hour="10",
        day_of_week="*",
        day_of_month="*",
        month_of_year="*",
        timezone="UTC",
    )
    model = PeriodicTask(
        crontab=crontab,
        name="daily_report",
        task="echo",
        enabled=True,
        total_run_count=0,
        last_run_at=last_run_at,
    )
    model.celery_options = {}

    with (
        patch.object(app, "now", return_value=LOAD_TIME),
        patch.object(TzAwareCrontab, "nowfunc", return_value=LOAD_TIME),
    ):
        yield ModelEntry(model, session_scope=lambda: None, app=app)


def test_last_run_at_of_never_run_task_defaults_to_load_time(app):  # noqa: ANN001, ANN201
    """Root cause: a NULL ``last_run_at`` is filled in with the *load* time.

    The task has never run, yet after being loaded into memory it reports
    having run at the very moment beat read it from the database
    (``ModelEntry.__init__`` -> ``self._default_now()``).
    """
    with loaded_entry(app) as entry:
        assert entry.last_run_at == LOAD_TIME
        # The default is written back to the model, so it will be persisted
        # on the next save()/commit as if it were a real run timestamp.
        assert entry.model.last_run_at == LOAD_TIME


def test_never_run_task_loaded_just_after_scheduled_time_is_silently_skipped(app):  # noqa: ANN001, ANN201
    """Bug demonstration: today's occurrence is lost and pushed ~24h into the future.

    Beat is alive 30 seconds after the 10:00 occurrence, the task never ran,
    and yet ``is_due`` reports "not due, check again in ~23h59m" -- the
    occurrence that just passed is silently rescheduled for tomorrow.
    """
    with loaded_entry(app) as entry:
        is_due, next_check_in = entry.is_due()

    assert is_due is False
    # Next check is tomorrow's 10:00 occurrence (86400s - the 30s already
    # elapsed), not today's missed one.
    assert next_check_in == pytest.approx(ONE_DAY - 30, abs=2)


def test_same_restart_with_a_previously_run_task_catches_up(app):  # noqa: ANN001, ANN201
    """Counterexample to "last_run_at does not matter for cron-style tasks".

    Identical schedule, identical restart timing; the ONLY difference is that
    ``last_run_at`` holds yesterday's run instead of being NULL.  Now
    ``TzAwareCrontab.is_due`` (the very function that supposedly makes
    ``last_run_at`` irrelevant) computes the next fire time FROM
    ``last_run_at``, sees that today's 10:00 already passed, and runs the
    task immediately.  Same restart, opposite outcome, driven purely by
    ``last_run_at`` -- which is what makes the never-run default a bug.
    """
    with loaded_entry(app, last_run_at=SCHEDULED_TIME - dt.timedelta(days=1)) as entry:
        is_due, _ = entry.is_due()

    assert is_due is True


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ModelEntry.__init__ initialises a never-run task's last_run_at to the "
        "load time instead of anchoring it to the task's creation time, so an "
        "occurrence falling between the scheduled time and the first tick is "
        "silently lost (see the two tests above)."
    ),
)
def test_expected_never_run_task_should_catch_up_like_a_previously_run_one(app):  # noqa: ANN001, ANN201
    """Expected behaviour once fixed: remove the xfail marker when it XPASSes.

    A never-run task whose occurrence just passed while beat was alive should
    be due, exactly like a previously-run task in the same situation.
    """
    with loaded_entry(app) as entry:
        is_due, _ = entry.is_due()

    assert is_due is True
