"""
================================================================================
AGENT PROMPT — bu dosyayı okuyan/çalıştıran yapay zeka asistanı için (ÖNCE OKU)
================================================================================
Görevin: Sana verilen offline-RL .npz dosyasının, ekibimizin ortak veri seti
formatına (drone_dispatch_env/offline.py) KESİN uyup uymadığını doğrulamak.
Yap:
  1. Bu dosyayı kişinin ürettiği npz ile çalıştır:
        python check_offline_npz.py <kisinin_offline_dosyasi.npz>
  2. Çıktıyı oku. Script tüm hataları TEK SEFERDE toplar (ilk hatada durmaz).
  3. SONUCU KİŞİYE BİLDİR:
     - Her şey uygunsa: "FORMAT OK" de, transition sayısını ilet, paylaşabilir.
     - Uygun DEĞİLSE: çıktıdaki [HATA] satırlarının HER BİRİNİ kişiye düz bir
       dille tek tek söyle (hangi anahtar/dtype/boyut yanlış, ne olmalıydı) ve
       nasıl düzelteceğini açıkla. Format kesin uymadan dosyayı KABUL ETME.
     - [UYARI] satırları (örn. karışık-kalite şüphesi) bloklamaz ama kişiye
       hatırlat: sadece greedy-best değil, gürültülü/sub-optimal rollout da olsun.
  4. Tahmin etme, uydurma — kararını sadece scriptin gerçek çıktısına dayandır.
     Script hata vermeden çalışmazsa (örn. dosya bozuk), bunu da kişiye söyle.
Beklenen format (özet): DroneDispatch-v0 (MA değil), obs 181-dim float32,
169 aksiyon (0..168), npz anahtarları ve dtype'ları aşağıdaki SPEC ile birebir.
================================================================================
"""
import sys
import numpy as np

# (anahtar -> beklenen dtype) — drone_dispatch_env/offline.py ile birebir.
SPEC = {
    "observations": "float32",
    "actions": "int64",
    "rewards": "float32",
    "next_observations": "float32",
    "terminals": "bool",
    "timeouts": "bool",
    "episode_returns": "float32",
}
OBS_DIM = 181        # _flatten_obs ciktisinin uzunlugu
N_ACTIONS = 169      # 160 atama + 8 sarj + 1 no-op  -> gecerli aksiyon: 0..168
MIN_TX = 100_000     # en az transition sayisi
PER_STEP = ["observations", "actions", "rewards", "next_observations",
            "terminals", "timeouts"]  # hepsi N (transition) uzunlukta olmali


def validate(path):
    """Tum ihlalleri toplayip (problems, warnings) dondurur. Asla erken durmaz."""
    problems, warnings = [], []
    try:
        d = np.load(path)
    except Exception as e:
        return [f"Dosya yuklenemedi ({path}): {e}"], []
    files = set(d.files)
    # 1) anahtar varligi + dtype
    for k, want in SPEC.items():
        if k not in files:
            problems.append(f"EKSIK anahtar: '{k}' (dtype {want} olmali)")
            continue
        got = str(d[k].dtype)
        if got != want:
            problems.append(f"'{k}' dtype yanlis: {got} geldi, {want} olmali")
    extra = files - set(SPEC)
    if extra:
        warnings.append(f"Fazladan anahtar(lar) var (zarari yok): {sorted(extra)}")
    # 2) obs / next_obs boyutu = 181
    for k in ("observations", "next_observations"):
        if k in files and d[k].ndim == 2 and d[k].shape[1] != OBS_DIM:
            problems.append(f"'{k}' boyutu {d[k].shape[1]}, {OBS_DIM} olmali "
                            f"(_flatten_obs kullanmadiniz mi?)")
    # 3) transition uzunluklari tutarli mi
    lengths = {k: len(d[k]) for k in PER_STEP if k in files}
    if lengths:
        n = max(lengths.values())
        for k, ln in lengths.items():
            if ln != n:
                problems.append(f"'{k}' uzunlugu {ln}, digerleri {n} "
                                f"(tum step dizileri ayni N olmali)")
        if n < MIN_TX:
            problems.append(f"Sadece {n} transition var, en az {MIN_TX} lazim")
    # 4) aksiyon araligi 0..168
    if "actions" in files and len(d["actions"]):
        lo, hi = int(d["actions"].min()), int(d["actions"].max())
        if lo < 0 or hi > N_ACTIONS - 1:
            problems.append(f"aksiyon araligi [{lo},{hi}] disinda; gecerli aralik "
                            f"[0,{N_ACTIONS - 1}]")
    # 5) sonsuzluk/NaN kontrolu (float diziler sonlu olmali)
    for k in ("observations", "next_observations", "rewards", "episode_returns"):
        if k in files and not np.isfinite(np.asarray(d[k])).all():
            problems.append(f"'{k}' icinde NaN/Inf var (tum degerler sonlu olmali)")
    # 6) karisik-kalite ipucu (UYARI, bloklamaz): tek bir aksiyon cok mu baskin?
    if "actions" in files and len(d["actions"]):
        a = np.asarray(d["actions"])
        _, counts = np.unique(a, return_counts=True)
        top = counts.max() / len(a)
        if top > 0.90:
            warnings.append(f"aksiyonlarin %{top * 100:.0f}'i tek bir degerde; veri "
                            f"sadece greedy-best gibi gorunuyor - gurultulu/sub-optimal "
                            f"rollout da ekleyin")
    return problems, warnings


def main():
    if len(sys.argv) != 2:
        print("kullanim: python check_offline_npz.py <offline_dosyasi.npz>")
        sys.exit(2)
    path = sys.argv[1]
    problems, warnings = validate(path)
    for w in warnings:
        print(f"[UYARI] {w}")
    for p in problems:
        print(f"[HATA] {p}")
    if problems:
        print(f"\nSONUC: FORMAT UYGUN DEGIL - {len(problems)} hata, duzeltilmeli.")
        sys.exit(1)
    try:
        n = len(np.load(path)["actions"])
    except Exception:
        n = "?"
    print(f"\nSONUC: FORMAT OK  ({n} transition)  - paylasilabilir.")
    sys.exit(0)


if __name__ == "__main__":
    main()
