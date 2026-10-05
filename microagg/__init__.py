"""
microagg - Micro-aggregation for privacy-preserving publication of income and
tax microdata.

Package layout
--------------
config          attribute roles, defaults and constants
tax             stylised progressive income-tax schedule
data_generator  reproducible synthetic income/tax population
preprocessing   input validation, edit rules and missing-value imputation
features        distance-space encoding (log, z-score, one-hot)
mdav            MDAV, individual ranking and random-partition algorithms
anonymizer      end-to-end anonymisation pipeline
release         release builder + "release guard" that blocks unsafe exports
utility         information-loss and analytical-validity metrics
privacy         disclosure-risk metrics and attack simulations
safe_display    helpers that coarsen original values before they reach a UI
experiments     experiment grid (k x method) and scalability benchmark
plotting        static matplotlib figures for the report
report_builder  fills docs/report_template.md with computed results
"""

__version__ = "1.0.0"
