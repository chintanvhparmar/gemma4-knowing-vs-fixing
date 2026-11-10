"""Build the post-cutoff task set in the same shape as the official dev tasks.

Output (in tasks_postcutoff/out/):
  tasks.jsonl            8 fields like the dev set: instance_id, repo, base_commit, patch,
                         test_patch, problem_statement, hints_text, created_at
  snapshots/<id>.tgz     the small repository BEFORE the fix
  meta.json              our extra info per task: library, old/new pinned versions, deps, probe id
Patches are unified diffs with '--- a/' / '+++ b/' headers and no 'diff --git' lines (the dev-set
format, verified 9 Nov). New test files use '--- /dev/null'.

Each task = a small project that worked on the OLD library version and breaks after upgrading to
the NEW version (latest release, checked on PyPI 9 Nov 2026). Which changes: the 12-task shortlist
from the knowledge probe (PLAN.md 8.0a; probe/results/hand_labels_*.json).

Snapshot format verified on Kaggle 9 Nov (fastapi_11194.tgz): a git repo (.git included), members
prefixed './' at the archive root. Ours match (make_git_snapshot); base_commit = the real commit hash.
Usage: python tasks_postcutoff/build_tasks.py
"""
from __future__ import annotations

import difflib
import io
import json
import tarfile
import textwrap
from pathlib import Path

OUT = Path(__file__).parent / "out"
CREATED_AT = "2026-11-09T00:00:00Z"
SNAPSHOT_TOP_DIR = ""  # "" = files at archive root. Set after inspecting an official snapshot.


def d(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def pyproject(name: str, pkg: str, deps: list[str]) -> str:
    dep_list = ", ".join(f'"{x}"' for x in deps)
    return d(f'''
        [build-system]
        requires = ["setuptools>=61"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{name}"
        version = "0.1.0"
        requires-python = ">=3.11"
        dependencies = [{dep_list}]

        [tool.setuptools]
        packages = ["{pkg}"]

        [tool.pytest.ini_options]
        pythonpath = ["."]
        testpaths = ["tests"]
        ''')


# --------------------------------------------------------------------------------------------
# Task specs. files = snapshot (before fix); fix = changed files after the reference fix;
# tests = hidden test files (become test_patch). old/new = exact pinned versions for validation.
# --------------------------------------------------------------------------------------------
SPECS: list[dict] = []

# 1 marshmallow 4.0: @validates methods receive data_key=
SPECS.append(dict(
    id="pc_marshmallow_validates", lib="marshmallow", probe="marshmallow_validates_data_key",
    old=["marshmallow==3.23.0"], new=["marshmallow==4.4.0"],
    files={
        "pyproject.toml": pyproject("userapi", "userapi", ["marshmallow"]),
        "README.md": "# userapi\n\nParses and validates user payloads.\n",
        "userapi/__init__.py": "",
        "userapi/schemas.py": d('''
            from marshmallow import Schema, ValidationError, fields, validates


            class UserSchema(Schema):
                name = fields.Str(required=True)
                age = fields.Int(required=True)

                @validates("age")
                def validate_age(self, value):
                    if value < 0:
                        raise ValidationError("Age must be non-negative.")
            '''),
        "userapi/service.py": d('''
            from .schemas import UserSchema


            def parse_user(payload: dict) -> dict:
                """Validate an incoming user payload and return the loaded data."""
                return UserSchema().load(payload)
            '''),
    },
    fix={"userapi/schemas.py": d('''
            from marshmallow import Schema, ValidationError, fields, validates


            class UserSchema(Schema):
                name = fields.Str(required=True)
                age = fields.Int(required=True)

                @validates("age")
                def validate_age(self, value, **kwargs):
                    if value < 0:
                        raise ValidationError("Age must be non-negative.")
            ''')},
    tests={"tests/test_parse_user.py": d('''
            import pytest
            from marshmallow import ValidationError

            from userapi.service import parse_user


            def test_valid_user_is_loaded():
                assert parse_user({"name": "Ada", "age": 36}) == {"name": "Ada", "age": 36}


            def test_negative_age_is_rejected():
                with pytest.raises(ValidationError) as exc:
                    parse_user({"name": "Bob", "age": -1})
                assert "age" in exc.value.messages
            ''')},
    problem=d('''
        parse_user() crashes for every payload after a dependency upgrade

        Since we upgraded our dependencies, every call to `userapi.service.parse_user()` fails with:

            TypeError: UserSchema.validate_age() got an unexpected keyword argument 'data_key'

        Before the upgrade, valid payloads were loaded and a negative `age` was rejected with a
        `ValidationError` on the `age` field. That is still the expected behaviour.
        '''),
))

# 2 marshmallow 4.0: Schema(context=) / self.context removed
SPECS.append(dict(
    id="pc_marshmallow_context", lib="marshmallow", probe="marshmallow_schema_context",
    old=["marshmallow==3.23.0"], new=["marshmallow==4.4.0"],
    files={
        "pyproject.toml": pyproject("articles", "articles", ["marshmallow"]),
        "README.md": "# articles\n\nSerialises articles for the web API.\n",
        "articles/__init__.py": "",
        "articles/schemas.py": d('''
            from marshmallow import Schema, fields


            class ArticleSchema(Schema):
                id = fields.Int()
                title = fields.Str()
                can_edit = fields.Method("get_can_edit")

                def get_can_edit(self, obj):
                    user = self.context.get("user")
                    return user is not None and user["id"] == obj["author_id"]
            '''),
        "articles/service.py": d('''
            from .schemas import ArticleSchema


            def serialize_article(article: dict, user: dict | None) -> dict:
                """Dump an article for the API; `can_edit` depends on the requesting user."""
                return ArticleSchema(context={"user": user}).dump(article)
            '''),
    },
    fix={
        "articles/schemas.py": d('''
            from marshmallow import Schema, fields
            from marshmallow.experimental.context import Context


            class ArticleSchema(Schema):
                id = fields.Int()
                title = fields.Str()
                can_edit = fields.Method("get_can_edit")

                def get_can_edit(self, obj):
                    user = Context.get({}).get("user")
                    return user is not None and user["id"] == obj["author_id"]
            '''),
        "articles/service.py": d('''
            from marshmallow.experimental.context import Context

            from .schemas import ArticleSchema


            def serialize_article(article: dict, user: dict | None) -> dict:
                """Dump an article for the API; `can_edit` depends on the requesting user."""
                with Context({"user": user}):
                    return ArticleSchema().dump(article)
            '''),
    },
    tests={"tests/test_serialize_article.py": d('''
            from articles.service import serialize_article

            ARTICLE = {"id": 7, "title": "Hello", "author_id": 1}


            def test_author_can_edit():
                assert serialize_article(ARTICLE, {"id": 1}) == {"id": 7, "title": "Hello", "can_edit": True}


            def test_other_user_cannot_edit():
                assert serialize_article(ARTICLE, {"id": 2})["can_edit"] is False


            def test_anonymous_cannot_edit():
                assert serialize_article(ARTICLE, None)["can_edit"] is False
            ''')},
    problem=d('''
        serialize_article() raises TypeError after upgrading dependencies

        After a dependency upgrade, `articles.service.serialize_article()` fails immediately:

            TypeError: Schema.__init__() got an unexpected keyword argument 'context'

        Expected: the article is dumped with `can_edit` = True only when the requesting user is the
        article's author (False for other users and for anonymous requests, `user=None`).
        '''),
))

# 3 polars 2.0: collect(streaming=True) removed -> engine="streaming"
SPECS.append(dict(
    id="pc_polars_streaming", lib="polars", probe="polars_streaming_engine",
    old=["polars==1.19.0"], new=["polars==2.1.0"],
    files={
        "pyproject.toml": pyproject("reports", "reports", ["polars"]),
        "README.md": "# reports\n\nSales report helpers on top of polars LazyFrames.\n",
        "reports/__init__.py": "",
        "reports/summary.py": d('''
            import polars as pl


            def total_by_region(lf: pl.LazyFrame) -> pl.DataFrame:
                """Total `amount` per `region`, sorted by region. Uses the streaming engine (large inputs)."""
                return (
                    lf.group_by("region")
                    .agg(pl.col("amount").sum().alias("total"))
                    .sort("region")
                    .collect(streaming=True)
                )
            '''),
    },
    fix={"reports/summary.py": d('''
            import polars as pl


            def total_by_region(lf: pl.LazyFrame) -> pl.DataFrame:
                """Total `amount` per `region`, sorted by region. Uses the streaming engine (large inputs)."""
                return (
                    lf.group_by("region")
                    .agg(pl.col("amount").sum().alias("total"))
                    .sort("region")
                    .collect(engine="streaming")
                )
            ''')},
    tests={"tests/test_summary.py": d('''
            import polars as pl

            from reports.summary import total_by_region


            def test_total_by_region():
                lf = pl.LazyFrame({"region": ["b", "a", "b", "a"], "amount": [1, 2, 3, 4]})
                out = total_by_region(lf)
                assert out.to_dict(as_series=False) == {"region": ["a", "b"], "total": [6, 4]}
            ''')},
    problem=d('''
        total_by_region() fails after upgrading polars

        After upgrading our dependencies, `reports.summary.total_by_region()` raises an error for any
        LazyFrame instead of returning the per-region totals. The report must still be computed with
        the streaming engine because production inputs are large.

        Expected: a DataFrame with columns `region` and `total`, sorted by region.
        '''),
))

# 4 click 8.2: command names drop the _cmd suffix
SPECS.append(dict(
    id="pc_click_command_suffix", lib="click", probe="click_command_suffix",
    old=["click==8.1.8"], new=["click==8.5.1"],
    files={
        "pyproject.toml": pyproject("projtool", "projtool", ["click"]),
        "README.md": d('''
            # projtool

            Usage:

                projtool init-cmd   # create the project skeleton
                projtool build      # build the project
            '''),
        "projtool/__init__.py": "",
        "projtool/cli.py": d('''
            import click


            @click.group()
            def cli():
                """Project tool."""


            @cli.command()
            def init_cmd():
                """Create the project skeleton."""
                click.echo("initialized")


            @cli.command()
            def build():
                """Build the project."""
                click.echo("built")
            '''),
    },
    fix={"projtool/cli.py": d('''
            import click


            @click.group()
            def cli():
                """Project tool."""


            @cli.command("init-cmd")
            def init_cmd():
                """Create the project skeleton."""
                click.echo("initialized")


            @cli.command()
            def build():
                """Build the project."""
                click.echo("built")
            ''')},
    tests={"tests/test_cli.py": d('''
            from click.testing import CliRunner

            from projtool.cli import cli


            def test_init_cmd_command():
                result = CliRunner().invoke(cli, ["init-cmd"])
                assert result.exit_code == 0, result.output
                assert "initialized" in result.output


            def test_build_command():
                result = CliRunner().invoke(cli, ["build"])
                assert result.exit_code == 0, result.output
                assert "built" in result.output
            ''')},
    problem=d('''
        `projtool init-cmd` stopped working after a dependency upgrade

        Our deploy scripts run `projtool init-cmd` (as documented in the README). After upgrading
        dependencies it fails with:

            Error: No such command 'init-cmd'. Did you mean 'init'?

        and exit code 2. `projtool build` still works. The documented command names must keep working.
        '''),
))

# 5 click 8.2: help shown by no_args_is_help exits with 2 instead of 0
SPECS.append(dict(
    id="pc_click_no_args_exit", lib="click", probe="click_no_args_is_help_exit",
    old=["click==8.1.8"], new=["click==8.5.1"],
    files={
        "pyproject.toml": pyproject("devtool", "devtool", ["click"]),
        "README.md": d('''
            # devtool

            Running `devtool` with no arguments prints the usage text and exits successfully
            (exit code 0); our CI smoke test relies on this.
            '''),
        "devtool/__init__.py": "",
        "devtool/cli.py": d('''
            import click


            @click.group()
            def cli():
                """Developer tool."""


            @cli.command()
            def build():
                """Build the project."""
                click.echo("built")
            '''),
    },
    fix={"devtool/cli.py": d('''
            import click


            @click.group(invoke_without_command=True)
            @click.pass_context
            def cli(ctx):
                """Developer tool."""
                if ctx.invoked_subcommand is None:
                    click.echo(ctx.get_help())


            @cli.command()
            def build():
                """Build the project."""
                click.echo("built")
            ''')},
    tests={"tests/test_cli.py": d('''
            from click.testing import CliRunner

            from devtool.cli import cli


            def test_no_args_prints_usage_and_succeeds():
                result = CliRunner().invoke(cli, [])
                assert result.exit_code == 0, result.output
                assert "Usage" in result.output
                assert "build" in result.output


            def test_build_command():
                result = CliRunner().invoke(cli, ["build"])
                assert result.exit_code == 0, result.output
                assert "built" in result.output
            ''')},
    problem=d('''
        CI smoke test fails: `devtool` with no arguments now exits with code 2

        Our CI smoke test runs `devtool` with no arguments and expects the usage text and exit code 0,
        as the README documents. After a dependency upgrade the usage text is still printed, but the
        process exits with code 2, so CI fails. `devtool build` still works.
        '''),
))

# 6 scikit-learn 1.8: AdaBoostClassifier(algorithm=) removed
SPECS.append(dict(
    id="pc_sklearn_adaboost", lib="scikit-learn", probe="sklearn_adaboost_algorithm",
    old=["scikit-learn==1.6.1"], new=["scikit-learn==1.9.1"],
    files={
        "pyproject.toml": pyproject("churnmodel", "churnmodel", ["scikit-learn"]),
        "README.md": "# churnmodel\n\nModel factory for the churn classifier.\n",
        "churnmodel/__init__.py": "",
        "churnmodel/factory.py": d('''
            from sklearn.ensemble import AdaBoostClassifier


            def make_booster(n_estimators: int = 100) -> AdaBoostClassifier:
                """Boosted decision stumps (SAMME), as used in production."""
                return AdaBoostClassifier(algorithm="SAMME", n_estimators=n_estimators, random_state=0)
            '''),
    },
    fix={"churnmodel/factory.py": d('''
            from sklearn.ensemble import AdaBoostClassifier


            def make_booster(n_estimators: int = 100) -> AdaBoostClassifier:
                """Boosted decision stumps (SAMME), as used in production."""
                return AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
            ''')},
    tests={"tests/test_factory.py": d('''
            from sklearn.datasets import make_classification

            from churnmodel.factory import make_booster


            def test_booster_fits_and_predicts():
                X, y = make_classification(n_samples=200, n_features=6, random_state=0)
                model = make_booster(n_estimators=50).fit(X, y)
                assert model.n_estimators == 50
                assert model.score(X, y) > 0.8
            ''')},
    problem=d('''
        make_booster() raises TypeError after upgrading scikit-learn

        After a dependency upgrade, `churnmodel.factory.make_booster()` fails before any training
        starts with a `TypeError` from the `AdaBoostClassifier` constructor. Expected: it returns an
        unfitted AdaBoost classifier with the requested number of estimators that trains as before.
        '''),
))

# 7 scikit-learn 1.8: LogisticRegression(multi_class=) removed
SPECS.append(dict(
    id="pc_sklearn_logreg_multiclass", lib="scikit-learn", probe="sklearn_logreg_multi_class",
    old=["scikit-learn==1.6.1"], new=["scikit-learn==1.9.1"],
    files={
        "pyproject.toml": pyproject("topicmodel", "topicmodel", ["scikit-learn"]),
        "README.md": "# topicmodel\n\nMulticlass topic classifier factory.\n",
        "topicmodel/__init__.py": "",
        "topicmodel/factory.py": d('''
            from sklearn.linear_model import LogisticRegression


            def make_classifier() -> LogisticRegression:
                """Multinomial (softmax) logistic regression over all topics."""
                return LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000)
            '''),
    },
    fix={"topicmodel/factory.py": d('''
            from sklearn.linear_model import LogisticRegression


            def make_classifier() -> LogisticRegression:
                """Multinomial (softmax) logistic regression over all topics."""
                return LogisticRegression(solver="lbfgs", max_iter=1000)
            ''')},
    tests={"tests/test_factory.py": d('''
            import numpy as np
            from sklearn.datasets import make_classification

            from topicmodel.factory import make_classifier


            def test_multiclass_probabilities():
                X, y = make_classification(n_samples=300, n_features=8, n_informative=5,
                                           n_classes=3, random_state=0)
                clf = make_classifier().fit(X, y)
                proba = clf.predict_proba(X)
                assert proba.shape == (300, 3)
                assert np.allclose(proba.sum(axis=1), 1.0)
                assert clf.score(X, y) > 0.7
            ''')},
    problem=d('''
        make_classifier() fails after upgrading scikit-learn

        Since the dependency upgrade, `topicmodel.factory.make_classifier()` raises a `TypeError` when
        constructing the `LogisticRegression`. Expected: a multinomial (softmax) classifier over all
        classes that fits and returns per-class probabilities summing to 1.
        '''),
))

# 8 Django 6.0: CheckConstraint(check=) -> condition=
SPECS.append(dict(
    id="pc_django_checkconstraint", lib="Django", probe="django_checkconstraint_condition",
    old=["Django==5.1.4"], new=["Django==6.1.3"],
    files={
        "pyproject.toml": pyproject("shop", "shop", ["Django"]),
        "README.md": "# shop\n\nProduct catalogue app.\n",
        "conftest.py": d('''
            import django
            from django.conf import settings


            def pytest_configure():
                if not settings.configured:
                    settings.configure(
                        INSTALLED_APPS=["shop"],
                        DATABASES={"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}},
                        DEFAULT_AUTO_FIELD="django.db.models.AutoField",
                        USE_TZ=True,
                    )
                    django.setup()
            '''),
        "shop/__init__.py": "",
        "shop/models.py": d('''
            from django.db import models


            class Product(models.Model):
                name = models.CharField(max_length=100)
                price = models.DecimalField(max_digits=10, decimal_places=2)

                class Meta:
                    app_label = "shop"
                    constraints = [
                        models.CheckConstraint(check=models.Q(price__gte=0), name="price_non_negative"),
                    ]
            '''),
    },
    fix={"shop/models.py": d('''
            from django.db import models


            class Product(models.Model):
                name = models.CharField(max_length=100)
                price = models.DecimalField(max_digits=10, decimal_places=2)

                class Meta:
                    app_label = "shop"
                    constraints = [
                        models.CheckConstraint(condition=models.Q(price__gte=0), name="price_non_negative"),
                    ]
            ''')},
    tests={"tests/test_product.py": d('''
            from decimal import Decimal

            import pytest
            from django.db import IntegrityError, connection, transaction

            from shop.models import Product


            @pytest.fixture(scope="module", autouse=True)
            def product_table():
                with connection.schema_editor() as editor:
                    editor.create_model(Product)
                yield


            def test_valid_price_is_saved():
                Product.objects.create(name="pen", price=Decimal("1.50"))
                assert Product.objects.filter(name="pen").exists()


            def test_negative_price_is_rejected():
                with pytest.raises(IntegrityError):
                    with transaction.atomic():
                        Product.objects.create(name="bad", price=Decimal("-1.00"))
            ''')},
    problem=d('''
        App fails to start after upgrading Django

        After a dependency upgrade, Django setup fails while importing `shop.models`:

            TypeError: CheckConstraint.__init__() got an unexpected keyword argument 'check'

        Expected: the app loads, products with a non-negative price can be saved, and the database
        still rejects negative prices.
        '''),
))

# 9 networkx 3.6: node_link_data default key "links" -> "edges"
SPECS.append(dict(
    id="pc_networkx_node_link", lib="networkx", probe="networkx_node_link_edges",
    old=["networkx==3.4.2"], new=["networkx==3.7"],
    files={
        "pyproject.toml": pyproject("graphapi", "graphapi", ["networkx"]),
        "README.md": "# graphapi\n\nJSON export of dependency graphs.\n",
        "graphapi/__init__.py": "",
        "graphapi/export.py": d('''
            import networkx as nx


            def edge_records(G: nx.Graph) -> list[dict]:
                """Edges as plain dicts (source, target, attributes) for the JSON API."""
                data = nx.node_link_data(G)
                return data["links"]


            def to_payload(G: nx.Graph) -> dict:
                """Compact payload: node ids and (source, target) pairs."""
                data = nx.node_link_data(G)
                return {
                    "nodes": [n["id"] for n in data["nodes"]],
                    "edges": [(e["source"], e["target"]) for e in data["links"]],
                }
            '''),
    },
    fix={"graphapi/export.py": d('''
            import networkx as nx


            def edge_records(G: nx.Graph) -> list[dict]:
                """Edges as plain dicts (source, target, attributes) for the JSON API."""
                data = nx.node_link_data(G, edges="links")
                return data["links"]


            def to_payload(G: nx.Graph) -> dict:
                """Compact payload: node ids and (source, target) pairs."""
                data = nx.node_link_data(G, edges="links")
                return {
                    "nodes": [n["id"] for n in data["nodes"]],
                    "edges": [(e["source"], e["target"]) for e in data["links"]],
                }
            ''')},
    tests={"tests/test_export.py": d('''
            import networkx as nx

            from graphapi.export import edge_records, to_payload


            def test_edge_records():
                G = nx.Graph()
                G.add_edge("a", "b", weight=2)
                assert edge_records(G) == [{"source": "a", "target": "b", "weight": 2}]


            def test_payload():
                G = nx.path_graph(3)
                assert to_payload(G) == {"nodes": [0, 1, 2], "edges": [(0, 1), (1, 2)]}
            ''')},
    problem=d('''
        Graph export fails with KeyError: 'links'

        After a dependency upgrade, both `graphapi.export.edge_records()` and `to_payload()` fail for
        every graph with:

            KeyError: 'links'

        Expected: `edge_records()` returns one dict per edge with `source`, `target` and the edge
        attributes; `to_payload()` returns the node ids and (source, target) pairs.
        '''),
))

# 10 pandas 3.0: datetimes parsed from strings default to [us] -> int64 epoch arithmetic 1000x off
SPECS.append(dict(
    id="pc_pandas_epoch_seconds", lib="pandas", probe="pandas_datetime_unit_us",
    old=["pandas==2.2.3"], new=["pandas==3.1.1"],
    files={
        "pyproject.toml": pyproject("etltime", "etltime", ["pandas"]),
        "README.md": "# etltime\n\nTimestamp helpers for the ETL pipeline.\n",
        "etltime/__init__.py": "",
        "etltime/epoch.py": d('''
            import pandas as pd


            def to_epoch_seconds(values) -> pd.Series:
                """ISO-8601 strings -> Unix epoch seconds as int64."""
                ts = pd.to_datetime(pd.Series(values), utc=True)
                return ts.astype("int64") // 10**9
            '''),
    },
    fix={"etltime/epoch.py": d('''
            import pandas as pd


            def to_epoch_seconds(values) -> pd.Series:
                """ISO-8601 strings -> Unix epoch seconds as int64."""
                ts = pd.to_datetime(pd.Series(values), utc=True)
                return ((ts - pd.Timestamp("1970-01-01", tz="UTC")) // pd.Timedelta(seconds=1)).astype("int64")
            ''')},
    tests={"tests/test_epoch.py": d('''
            from etltime.epoch import to_epoch_seconds


            def test_epoch_seconds():
                out = to_epoch_seconds(["1970-01-01T00:00:00Z", "2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"])
                assert out.tolist() == [0, 1704067200, 1704067201]
                assert str(out.dtype) == "int64"
            ''')},
    problem=d('''
        to_epoch_seconds() returns values 1000x too small after upgrading pandas

        Since a dependency upgrade, `etltime.epoch.to_epoch_seconds(["2024-01-01T00:00:00Z"])` returns
        1704067 instead of 1704067200. No error is raised, so downstream jobs silently got wrong
        timestamps. Expected: Unix epoch seconds as int64.
        '''),
))

# 11 pandas 3.0: zoneinfo default instead of pytz -> tz.zone attribute missing
SPECS.append(dict(
    id="pc_pandas_tz_zone", lib="pandas", probe="pandas_zoneinfo_default",
    old=["pandas==2.2.3"], new=["pandas==3.1.1"],
    files={
        "pyproject.toml": pyproject("etlzones", "etlzones", ["pandas"]),
        "README.md": "# etlzones\n\nTime-zone helpers for reports.\n",
        "etlzones/__init__.py": "",
        "etlzones/zones.py": d('''
            import pandas as pd


            def zone_name(ts: pd.Timestamp) -> str:
                """IANA zone name of a tz-aware Timestamp, e.g. 'Europe/Berlin'."""
                return ts.tz.zone


            def label(ts: pd.Timestamp) -> str:
                """Report label like '2024-06-01 12:00 (Europe/Berlin)'."""
                return f"{ts:%Y-%m-%d %H:%M} ({zone_name(ts)})"
            '''),
    },
    fix={"etlzones/zones.py": d('''
            import pandas as pd


            def zone_name(ts: pd.Timestamp) -> str:
                """IANA zone name of a tz-aware Timestamp, e.g. 'Europe/Berlin'."""
                return str(ts.tz)


            def label(ts: pd.Timestamp) -> str:
                """Report label like '2024-06-01 12:00 (Europe/Berlin)'."""
                return f"{ts:%Y-%m-%d %H:%M} ({zone_name(ts)})"
            ''')},
    tests={"tests/test_zones.py": d('''
            import pandas as pd

            from etlzones.zones import label, zone_name


            def test_zone_name():
                assert zone_name(pd.Timestamp("2024-06-01 12:00", tz="Europe/Berlin")) == "Europe/Berlin"


            def test_label():
                ts = pd.Timestamp("2024-06-01 12:00", tz="America/New_York")
                assert label(ts) == "2024-06-01 12:00 (America/New_York)"
            ''')},
    problem=d('''
        zone_name() raises AttributeError after upgrading pandas

        After a dependency upgrade, `etlzones.zones.zone_name()` (and therefore `label()`) fails for
        tz-aware Timestamps with:

            AttributeError: 'zoneinfo.ZoneInfo' object has no attribute 'zone'

        Expected: the IANA zone name, e.g. 'Europe/Berlin'.
        '''),
))

# 12 Starlette 1.0: on_startup/on_shutdown removed -> lifespan
SPECS.append(dict(
    id="pc_starlette_lifespan", lib="starlette", probe="starlette_lifespan",
    old=["starlette==0.41.3", "httpx==0.28.1"], new=["starlette==1.10.0", "httpx==0.28.1"],
    files={
        "pyproject.toml": pyproject("healthapp", "healthapp", ["starlette", "httpx"]),
        "README.md": "# healthapp\n\nTiny Starlette service with a database connection.\n",
        "healthapp/__init__.py": "",
        "healthapp/db.py": d('''
            class Database:
                def __init__(self):
                    self.connected = False

                async def connect(self):
                    self.connected = True

                async def disconnect(self):
                    self.connected = False
            '''),
        "healthapp/main.py": d('''
            from starlette.applications import Starlette
            from starlette.responses import JSONResponse
            from starlette.routing import Route

            from .db import Database

            db = Database()


            async def health(request):
                return JSONResponse({"db": db.connected})


            async def startup():
                await db.connect()


            async def shutdown():
                await db.disconnect()


            app = Starlette(routes=[Route("/health", health)], on_startup=[startup], on_shutdown=[shutdown])
            '''),
    },
    fix={"healthapp/main.py": d('''
            import contextlib

            from starlette.applications import Starlette
            from starlette.responses import JSONResponse
            from starlette.routing import Route

            from .db import Database

            db = Database()


            async def health(request):
                return JSONResponse({"db": db.connected})


            @contextlib.asynccontextmanager
            async def lifespan(app):
                await db.connect()
                yield
                await db.disconnect()


            app = Starlette(routes=[Route("/health", health)], lifespan=lifespan)
            ''')},
    tests={"tests/test_health.py": d('''
            from starlette.testclient import TestClient

            from healthapp.main import app, db


            def test_db_connected_during_app_lifetime():
                with TestClient(app) as client:
                    assert client.get("/health").json() == {"db": True}
                assert db.connected is False
            ''')},
    problem=d('''
        Service no longer starts after upgrading Starlette

        After a dependency upgrade, importing `healthapp.main` fails with:

            TypeError: Starlette.__init__() got an unexpected keyword argument 'on_startup'

        Expected: the database connects when the app starts, `/health` reports `{"db": true}` while
        the app is running, and the database disconnects at shutdown.
        '''),
))


# --------------------------------------------------------------------------------------------
def unified(path: str, before: str | None, after: str) -> str:
    a = [] if before is None else before.splitlines(keepends=True)
    b = after.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile="/dev/null" if before is None else f"a/{path}",
                                tofile=f"b/{path}", n=3)
    return "".join(diff)


GIT_ENV = {  # fixed identity + dates -> reproducible commit hashes
    "GIT_AUTHOR_NAME": "postcutoff-builder", "GIT_AUTHOR_EMAIL": "builder@example.invalid",
    "GIT_COMMITTER_NAME": "postcutoff-builder", "GIT_COMMITTER_EMAIL": "builder@example.invalid",
    "GIT_AUTHOR_DATE": "2026-11-09T00:00:00+0000", "GIT_COMMITTER_DATE": "2026-11-09T00:00:00+0000",
}


def _git() -> str:
    import shutil
    g = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
    if not Path(g).exists() and not shutil.which("git"):
        raise RuntimeError("git not found (needed: official snapshots are git repos)")
    return g


def make_git_snapshot(files: dict[str, str], task_id: str) -> tuple[bytes, str]:
    """Snapshot like the official ones (verified 9 Nov): a git repo, archive members prefixed './'.

    The harness itself commits the workspace and tags `_swegemma_baseline` at setup, and
    submit_patch runs `git diff --binary _swegemma_baseline` (swegemma container_setup.py:578,
    tools/execution.py:86-89), so the workspace must be a git repository. base_commit isn't checked out.
    """
    import os
    import subprocess
    import tempfile

    g = _git()
    env = {**os.environ, **GIT_ENV}
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / task_id
        for rel, text in files.items():
            p = repo / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(text.encode("utf-8"))  # LF line endings, no BOM
        run = lambda *a: subprocess.run([g, *a], cwd=repo, env=env, check=True, capture_output=True, text=True)
        run("init", "-q", "-b", "main")
        run("config", "core.autocrlf", "false")
        run("config", "core.fileMode", "false")
        run("add", "-A")
        run("commit", "-q", "-m", f"{task_id}: snapshot before the dependency fix")
        commit = run("rev-parse", "HEAD").stdout.strip()
        # LF check on raw BYTES (text mode would hide CRLF): git must not have rewritten any file
        for rel, text in files.items():
            blob = subprocess.run([g, "show", f"HEAD:{rel}"], cwd=repo, env=env, check=True,
                                  capture_output=True).stdout
            assert blob == text.encode("utf-8"), f"{task_id}: {rel} changed in git (line endings?)"
            assert b"\r" not in (repo / rel).read_bytes(), f"{task_id}: CR in working file {rel}"
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            def norm(ti):
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mtime = 0
                return ti
            tar.add(repo, arcname=".", filter=norm)
    return buf.getvalue(), commit


def main() -> int:
    (OUT / "snapshots").mkdir(parents=True, exist_ok=True)
    rows, meta = [], {}
    for s in SPECS:
        for p in s["fix"]:
            assert p in s["files"], f"{s['id']}: fix touches unknown file {p}"
        for p in s["tests"]:
            assert p not in s["files"], f"{s['id']}: hidden test {p} must not be in the snapshot"
        patch = "".join(unified(p, s["files"][p], s["fix"][p]) for p in sorted(s["fix"]))
        test_patch = "".join(unified(p, None, s["tests"][p]) for p in sorted(s["tests"]))
        assert patch and test_patch, s["id"]
        tgz, commit = make_git_snapshot(s["files"], s["id"])
        (OUT / "snapshots" / f"{s['id']}.tgz").write_bytes(tgz)
        rows.append(dict(instance_id=s["id"], repo=f"postcutoff/{s['lib'].lower()}",
                         base_commit=commit, patch=patch, test_patch=test_patch,
                         problem_statement=s["problem"], hints_text="", created_at=CREATED_AT))
        meta[s["id"]] = dict(lib=s["lib"], probe=s["probe"], old=s["old"], new=s["new"],
                             test_files=sorted(s["tests"]))
    (OUT / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"built {len(rows)} tasks -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
