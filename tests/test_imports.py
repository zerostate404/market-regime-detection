"""Basic smoke test to ensure the package is installed correctly."""


def test_mrd_imports():
    import mrd
    from mrd.phase1_pipeline import run_phase1
    from mrd.phase2_features import run_all_features
    from mrd.phase3_fusion import run_fusion
    from mrd.phase4_backtest import run_phase4
    from mrd.phase5_live import run_live_update

    assert mrd.__version__ == "0.1.0"
