"""pilot + estimate + one_day scheduler dry run with tiny settings (CPU)."""
import sys, os, time, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
C.CHAINS, C.WARMUP, C.SAMPLES = 2, 40, 40
C.REFIT = (40, 0.99, 10, 80)
C.RHAT_MAX, C.ESS_MIN = 1.5, 5
C.SIZES = ((10, 60),)
C.REF = dict(alpha="2", mu="0.2", regime="gradual", I=10, T=60)
OUT = os.path.join(os.environ.get("SP", "/tmp"), "test_pilot")
import run
log = run.make_logger(OUT) if os.path.isdir(OUT) or not os.makedirs(OUT, exist_ok=True) else None
rep = run.pilot(OUT, 4, sizes=(1, 2))
print(json.dumps(rep["model"]))
print(run.estimate(OUT))
print(run.progress(OUT))
