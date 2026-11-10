"""Silent-change task PROTOTYPES (10 Nov). Same format and helpers as tasks_postcutoff/build_tasks.py,
written to tasks_silent/out/ so the frozen 12-task set is untouched.

A silent change = code written for the old library version runs on the new version without any error
or warning but gives a different result. Each prototype must pass, before it is kept:
  1. knowledge probe (round 3, probe set 3): the model does not know the change;
  2. validation at the exact pins (old no-fix PASS, new no-fix FAIL, new fix PASS), Python 3.12;
  3. a C0 pilot (2 runs): the baseline agent fails or clearly struggles (the lesson of 10 Nov).
Issue texts describe the symptom in user terms only (no library function named as the cause).
New-version pins = the versions already bundled in tasks_postcutoff/out/wheels.

    python -I tasks_silent/build_silent.py            -> tasks_silent/out/        (informative issues)
    python -I tasks_silent/build_silent.py --vague    -> tasks_silent/out_vague/  (same tasks, vague issues)
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
_spec = importlib.util.spec_from_file_location("build_tasks", HERE.parent / "tasks_postcutoff" / "build_tasks.py")
bt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bt)
d, pyproject, unified, make_git_snapshot = bt.d, bt.pyproject, bt.unified, bt.make_git_snapshot
CREATED_AT = "2026-11-10T00:00:00Z"

SPECS: list[dict] = []

# S1 pandas 3.0: groupby(observed=False) with several keys -> unobserved all()/any()/prod() = True/False/1 (was NaN)
SPECS.append(dict(
    id="sc_pandas_groupby_unobserved", lib="pandas", probe="pandas_groupby_observed_multikey_prod",
    old=["pandas==2.2.3"], new=["pandas==3.1.1"],
    files={
        "pyproject.toml": pyproject("shipreport", "shipreport", ["pandas"]),
        "README.md": "# shipreport\n\nDaily shipping status per region and product line.\n",
        "shipreport/__init__.py": "",
        "shipreport/status.py": d('''
            import pandas as pd

            REGIONS = ["north", "south", "east", "west"]
            PRODUCTS = ["basic", "pro", "enterprise"]


            def fully_shipped(orders: pd.DataFrame) -> dict:
                """For every (region, product) pair that HAS orders, report whether all its orders shipped.

                `orders` has columns region, product, shipped (bool). Pairs without orders are not reported.
                """
                df = orders.assign(
                    region=pd.Categorical(orders["region"], categories=REGIONS),
                    product=pd.Categorical(orders["product"], categories=PRODUCTS),
                )
                status = df.groupby(["region", "product"], observed=False)["shipped"].all()
                status = status.dropna()
                return {(r, p): bool(v) for (r, p), v in status.items()}
            '''),
    },
    fix={"shipreport/status.py": d('''
            import pandas as pd

            REGIONS = ["north", "south", "east", "west"]
            PRODUCTS = ["basic", "pro", "enterprise"]


            def fully_shipped(orders: pd.DataFrame) -> dict:
                """For every (region, product) pair that HAS orders, report whether all its orders shipped.

                `orders` has columns region, product, shipped (bool). Pairs without orders are not reported.
                """
                df = orders.assign(
                    region=pd.Categorical(orders["region"], categories=REGIONS),
                    product=pd.Categorical(orders["product"], categories=PRODUCTS),
                )
                status = df.groupby(["region", "product"], observed=True)["shipped"].all()
                return {(r, p): bool(v) for (r, p), v in status.items()}
            ''')},
    tests={"tests/test_status.py": d('''
            import pandas as pd

            from shipreport.status import fully_shipped


            def test_only_pairs_with_orders_are_reported():
                orders = pd.DataFrame({
                    "region": ["north", "north", "south", "east"],
                    "product": ["basic", "basic", "pro", "basic"],
                    "shipped": [True, False, True, True],
                })
                assert fully_shipped(orders) == {
                    ("north", "basic"): False,
                    ("south", "pro"): True,
                    ("east", "basic"): True,
                }


            def test_single_pair():
                orders = pd.DataFrame({"region": ["west"], "product": ["enterprise"], "shipped": [False]})
                assert fully_shipped(orders) == {("west", "enterprise"): False}
            ''')},
    problem=d('''
        Shipping report lists product lines that had no orders at all

        After a dependency upgrade, the daily shipping report (`shipreport.status.fully_shipped()`) is
        full of region/product pairs that had no orders that day, and all of them are shown as
        "fully shipped". For example, with orders only for north/basic, south/pro and east/basic, the
        report now contains 12 entries instead of 3.

        Expected (as before the upgrade and as the docstring says): only pairs that actually have
        orders are reported, with True if all their orders shipped and False otherwise.
        '''),
))

# S2 scikit-learn 1.8: median_absolute_error(sample_weight=...) uses the averaged weighted percentile
SPECS.append(dict(
    id="sc_sklearn_weighted_median_ae", lib="scikit-learn", probe="sklearn_median_ae_weighted",
    old=["scikit-learn==1.6.1"], new=["scikit-learn==1.9.1"],
    files={
        "pyproject.toml": pyproject("forecastkpi", "forecastkpi", ["scikit-learn"]),
        "README.md": "# forecastkpi\n\nWeekly forecast accuracy KPI (demand-weighted median absolute error).\n",
        "forecastkpi/__init__.py": "",
        "forecastkpi/kpi.py": d('''
            from sklearn.metrics import median_absolute_error


            def weekly_kpi(actual, forecast, demand_weight):
                """Demand-weighted median absolute forecast error for one week (lower is better)."""
                return float(median_absolute_error(actual, forecast, sample_weight=demand_weight))
            '''),
    },
    fix={"forecastkpi/kpi.py": d('''
            import numpy as np


            def weekly_kpi(actual, forecast, demand_weight):
                """Demand-weighted median absolute forecast error for one week (lower is better).

                Weighted lower median: the smallest absolute error e such that the errors <= e carry at
                least half of the total weight (the definition used by all historical reports).
                """
                err = np.abs(np.asarray(actual, dtype=float) - np.asarray(forecast, dtype=float))
                w = np.asarray(demand_weight, dtype=float)
                order = np.argsort(err, kind="stable")
                cum = np.cumsum(w[order])
                idx = int(np.searchsorted(cum, 0.5 * cum[-1], side="left"))
                return float(err[order][idx])
            ''')},
    tests={"tests/test_kpi.py": d('''
            import pytest

            from forecastkpi.kpi import weekly_kpi


            def test_kpi_matches_historical_reports():
                assert weekly_kpi([0, 0, 0, 0], [1, 2, 3, 4], [1, 1, 1, 1]) == pytest.approx(2.0)


            def test_kpi_uses_weights():
                assert weekly_kpi([0, 0, 0, 0], [1, 2, 3, 4], [1, 1, 1, 5]) == pytest.approx(4.0)
                assert weekly_kpi([10, 10, 10], [12, 9, 16], [2, 2, 1]) == pytest.approx(2.0)
            ''')},
    problem=d('''
        Weekly forecast KPI no longer matches the historical reports

        After a dependency upgrade, `forecastkpi.kpi.weekly_kpi()` returns different values than before
        for the same data, so this year's numbers are no longer comparable with the archived reports.
        Our regression fixture (actual [0, 0, 0, 0], forecast [1, 2, 3, 4], equal weights) used to give
        2.0 and now gives 2.5. Nothing in our code changed.

        Expected: the KPI must keep returning exactly the values it returned before the upgrade.
        '''),
))

# S3 click 8.2+: an envvar on a flag with flag_value is no longer taken as the option's value
SPECS.append(dict(
    id="sc_click_flag_envvar", lib="click", probe="click_flag_value_envvar",
    old=["click==8.1.8"], new=["click==8.5.1"],
    files={
        "pyproject.toml": pyproject("shipit", "shipit", ["click"]),
        "README.md": d('''
            # shipit

            `shipit deploy` deploys in `safe` mode. `shipit deploy --fast` uses `fast` mode.
            In CI the mode can also be set with the environment variable `SHIPIT_MODE`
            (e.g. `SHIPIT_MODE=canary shipit deploy`).
            '''),
        "shipit/__init__.py": "",
        "shipit/cli.py": d('''
            import click


            @click.group()
            def cli():
                pass


            @cli.command()
            @click.option("--fast", "mode", is_flag=True, flag_value="fast", envvar="SHIPIT_MODE")
            def deploy(mode):
                """Deploy the current build."""
                click.echo(f"deploying in {mode or 'safe'} mode")
            '''),
    },
    fix={"shipit/cli.py": d('''
            import click


            @click.group()
            def cli():
                pass


            @cli.command()
            @click.option("--fast", "fast", is_flag=True, help="Use fast mode.")
            @click.option("--mode", "mode", default="safe", envvar="SHIPIT_MODE", hidden=True)
            def deploy(fast, mode):
                """Deploy the current build."""
                if fast:
                    mode = "fast"
                click.echo(f"deploying in {mode} mode")
            ''')},
    tests={"tests/test_cli.py": d('''
            from click.testing import CliRunner

            from shipit.cli import cli


            def run(args, env=None):
                r = CliRunner().invoke(cli, args, env=env or {})
                assert r.exit_code == 0, r.output
                return r.output.strip()


            def test_default_is_safe():
                assert run(["deploy"], env={"SHIPIT_MODE": None}) == "deploying in safe mode"


            def test_flag_selects_fast():
                assert run(["deploy", "--fast"], env={"SHIPIT_MODE": None}) == "deploying in fast mode"


            def test_env_var_sets_mode():
                assert run(["deploy"], env={"SHIPIT_MODE": "canary"}) == "deploying in canary mode"
            ''')},
    problem=d('''
        SHIPIT_MODE is ignored in CI after a dependency upgrade

        Our CI pipeline runs `SHIPIT_MODE=canary shipit deploy` (as documented in the README) to do
        canary deploys. Since a dependency upgrade the command prints a different mode and does not
        deploy in canary mode any more. Running locally with `--fast` or without anything still looks fine.

        Expected: `SHIPIT_MODE=<mode>` selects that mode, `--fast` selects fast, otherwise safe.
        '''),
))

# S4 pandas 3.0: Float64 arithmetic that produced NaN now produces NA
SPECS.append(dict(
    id="sc_pandas_float64_na", lib="pandas", probe="pandas_float64_nan_is_na",
    old=["pandas==2.2.3"], new=["pandas==3.1.1"],
    files={
        "pyproject.toml": pyproject("qualitystats", "qualitystats", ["pandas"]),
        "README.md": "# qualitystats\n\nData-quality counters for the sensor feed.\n",
        "qualitystats/__init__.py": "",
        "qualitystats/counters.py": d('''
            import numpy as np
            import pandas as pd


            def quality_counts(df: pd.DataFrame) -> dict:
                """Counters for the feed.

                `df` has nullable Float64 columns `reading` and `baseline` (a missing value = no data).
                - missing: rows where reading or baseline is missing
                - undefined_ratio: rows with data whose ratio reading/baseline is undefined (0/0)
                """
                ratio = df["reading"] / df["baseline"]
                missing = int(ratio.isna().sum())  # NA only where an input is missing
                undefined = int(np.isnan(ratio.to_numpy(dtype="float64", na_value=0.0)).sum())  # NaN from 0/0
                return {"missing": missing, "undefined_ratio": undefined}
            '''),
    },
    fix={"qualitystats/counters.py": d('''
            import pandas as pd


            def quality_counts(df: pd.DataFrame) -> dict:
                """Counters for the feed.

                `df` has nullable Float64 columns `reading` and `baseline` (a missing value = no data).
                - missing: rows where reading or baseline is missing
                - undefined_ratio: rows with data whose ratio reading/baseline is undefined (0/0)
                """
                has_data = df["reading"].notna() & df["baseline"].notna()
                zero_by_zero = has_data & (df["reading"] == 0) & (df["baseline"] == 0)
                return {"missing": int((~has_data).sum()), "undefined_ratio": int(zero_by_zero.fillna(False).sum())}
            ''')},
    tests={"tests/test_counters.py": d('''
            import pandas as pd

            from qualitystats.counters import quality_counts


            def frame(reading, baseline):
                return pd.DataFrame({"reading": pd.array(reading, dtype="Float64"),
                                     "baseline": pd.array(baseline, dtype="Float64")})


            def test_zero_by_zero_counts_as_undefined_not_missing():
                df = frame([0.0, 1.0, None, 2.0], [0.0, 2.0, 1.0, 4.0])
                assert quality_counts(df) == {"missing": 1, "undefined_ratio": 1}


            def test_clean_feed():
                df = frame([1.0, 2.0], [1.0, 4.0])
                assert quality_counts(df) == {"missing": 0, "undefined_ratio": 0}
            ''')},
    problem=d('''
        Sensor quality dashboard: "undefined ratio" is always 0 and "missing" is too high after upgrade

        Since a dependency upgrade, `qualitystats.counters.quality_counts()` always reports
        `undefined_ratio = 0`, even on days where we know sensors reported 0 against a 0 baseline,
        and those rows now show up in `missing` instead. Example: readings [0, 1, missing, 2] with
        baselines [0, 2, 1, 4] used to give missing=1, undefined_ratio=1; now missing=2, undefined_ratio=0.

        Expected: the counters return exactly what they returned before the upgrade.
        '''),
))


# VAGUE variants (10 Nov): same snapshot, fix and hidden tests; the issue says THAT behaviour changed after
# the upgrade but gives no example values, so the agent cannot fix by matching a target number. Modelled on
# the style of real upgrade reports (e.g. click #3111, "Error when updating click from 8.2.1 to 8.3.0").
# The repo's own README/docstrings still document the intended behaviour, as in a real project.
VAGUE = {
    "sc_pandas_groupby_unobserved": d('''
        Shipping report is wrong after a dependency upgrade

        The daily report produced by `shipreport.status.fully_shipped()` changed after we upgraded our
        dependencies; the operations team says it now contains entries it should not. Our code did not
        change. Expected: the report behaves as it did before the upgrade (see the function's docstring).
        '''),
    "sc_sklearn_weighted_median_ae": d('''
        Weekly forecast KPI no longer matches the historical reports

        After a dependency upgrade, `forecastkpi.kpi.weekly_kpi()` returns different values than before
        for the same data, so this year's numbers can't be compared with the archived reports. Nothing in
        our code changed. Expected: the KPI must keep returning exactly the values it returned before the upgrade.
        '''),
    "sc_click_flag_envvar": d('''
        CI deploys no longer run in the expected mode after a dependency upgrade

        Since a dependency upgrade, our CI deploys (configured as described in the README) don't run in
        the mode the pipeline asks for any more. Local runs look fine. Expected: `shipit deploy` behaves
        as documented in the README.
        '''),
    "sc_pandas_float64_na": d('''
        Sensor quality counters changed after a dependency upgrade

        After upgrading our dependencies, the counters returned by `qualitystats.counters.quality_counts()`
        no longer agree with last month's dashboards for the same raw data. Our code did not change.
        Expected: the counters return exactly what they returned before the upgrade (definitions in the docstring).
        '''),
}


# NO-SPEC variants (10 Nov): the expected (old) behaviour is written down NOWHERE: no example values in the
# issue, and the sentences/comments in docstrings and README that define it are removed. Only the old
# library version "knew" the right behaviour (the vague pilot showed C0 failing only in that situation).
# Same hidden tests and reference fix. sklearn already had no spec, so only its issue wording is reused.
NOSPEC_FILES = {
    "sc_pandas_groupby_unobserved": {"shipreport/status.py": d('''
        import pandas as pd

        REGIONS = ["north", "south", "east", "west"]
        PRODUCTS = ["basic", "pro", "enterprise"]


        def fully_shipped(orders: pd.DataFrame) -> dict:
            """Shipping status per (region, product) for the daily report."""
            df = orders.assign(
                region=pd.Categorical(orders["region"], categories=REGIONS),
                product=pd.Categorical(orders["product"], categories=PRODUCTS),
            )
            status = df.groupby(["region", "product"], observed=False)["shipped"].all()
            status = status.dropna()
            return {(r, p): bool(v) for (r, p), v in status.items()}
        ''')},
    "sc_click_flag_envvar": {"README.md": "# shipit\n\n`shipit deploy` deploys the current build.\n"},
    "sc_pandas_float64_na": {"qualitystats/counters.py": d('''
        import numpy as np
        import pandas as pd


        def quality_counts(df: pd.DataFrame) -> dict:
            """Data-quality counters for the sensor feed."""
            ratio = df["reading"] / df["baseline"]
            missing = int(ratio.isna().sum())
            undefined = int(np.isnan(ratio.to_numpy(dtype="float64", na_value=0.0)).sum())
            return {"missing": missing, "undefined_ratio": undefined}
        ''')},
}
NOSPEC_PROBLEM = {
    "sc_pandas_groupby_unobserved": d('''
        Shipping report is wrong after a dependency upgrade

        The daily report produced by `shipreport.status.fully_shipped()` changed after we upgraded our
        dependencies, and the operations team says it is wrong now. Our code did not change.
        Expected: the report is exactly as it was before the upgrade.
        '''),
    "sc_sklearn_weighted_median_ae": VAGUE["sc_sklearn_weighted_median_ae"],
    "sc_click_flag_envvar": d('''
        CI deploys no longer run in the requested mode after a dependency upgrade

        Our CI pipeline selects the deploy mode through the environment. Since a dependency upgrade,
        CI deploys don't run in the mode the pipeline asks for any more; local runs look fine.
        Expected: `shipit deploy` behaves exactly as it did before the upgrade.
        '''),
    "sc_pandas_float64_na": d('''
        Sensor quality counters changed after a dependency upgrade

        After upgrading our dependencies, the counters returned by `qualitystats.counters.quality_counts()`
        no longer agree with last month's dashboards for the same raw data. Our code did not change.
        Expected: the counters return exactly what they returned before the upgrade.
        '''),
}


def main() -> int:
    import sys

    global OUT
    vague = "--vague" in sys.argv
    nospec = "--nospec" in sys.argv
    if vague:
        OUT = HERE / "out_vague"
    if nospec:
        OUT = HERE / "out_nospec"
    (OUT / "snapshots").mkdir(parents=True, exist_ok=True)
    rows, meta = [], {}
    for s in SPECS:
        if vague:
            s = dict(s, id=s["id"] + "_vague", problem=VAGUE[s["id"]])
        if nospec:
            files = {**s["files"], **NOSPEC_FILES.get(s["id"], {})}
            s = dict(s, id=s["id"] + "_nospec", files=files, problem=NOSPEC_PROBLEM[s["id"]])
        for p in s["fix"]:
            assert p in s["files"], f"{s['id']}: fix touches unknown file {p}"
        for p in s["tests"]:
            assert p not in s["files"], f"{s['id']}: hidden test {p} must not be in the snapshot"
        patch = "".join(unified(p, s["files"][p], s["fix"][p]) for p in sorted(s["fix"]))
        test_patch = "".join(unified(p, None, s["tests"][p]) for p in sorted(s["tests"]))
        tgz, commit = make_git_snapshot(s["files"], s["id"])
        (OUT / "snapshots" / f"{s['id']}.tgz").write_bytes(tgz)
        rows.append(dict(instance_id=s["id"], repo=f"postcutoff/{s['lib'].lower()}", base_commit=commit,
                         patch=patch, test_patch=test_patch, problem_statement=s["problem"], hints_text="",
                         created_at=CREATED_AT))
        meta[s["id"]] = dict(lib=s["lib"], probe=s["probe"], old=s["old"], new=s["new"], test_files=sorted(s["tests"]))
    (OUT / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {len(rows)} silent prototypes -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
