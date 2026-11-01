#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
graybox_engine.py — "Café Poliçe" graybox (beyaz-kutu GLM) + metaheuristik (GA) saf prim motoru.

NEDİR
  Sigorta fiyatlamasının klasik "graybox" tarifi: yorumlanabilir bir yapı (white-box
  GLM) + öğrenilebilir ama opak bir parça (GA ile hiperparametre arama). Burada:
    * frekans  -> sklearn PoissonRegressor  (log-link)   [aktüeryal standart]
    * şiddet   -> sklearn GammaRegressor    (log-link)   [aktüeryal standart]
    * saf prim  = frekans_orani x beklenen_siddet
    * metaheuristik -> Genetik Algoritma (GA) ile iki GLM'in alpha (regularization)
      hiperparametresini GRID yerine arar (validation üzerinde deviance minimize).

VERİ (dürüst etiketleme)
  * freMTPL2 (açık, CC0) indirilebilirse frekans tarafında kullanılır; şiddet tarafı
    freMTPL2'de ayrı ve eksik olduğundan SENTETİK olarak üretilir ve ETİKETLENİR.
  * İndirilemezse tüm veri SENTETİK DEMO'dur (sabit seed) — ASLA gerçekmiş gibi
    sunulmaz. Veri kaynağı her çalıştırmada `veri` alanında yazılır.

KURAL: C: sürücüsüne YAZILMAZ. Çıktı (varsa) bu deponun kendi dizinindedir.

Kullanım:
  python src/graybox_engine.py            -> tam çalıştırma + özet
  python src/graybox_engine.py --test     -> kendini doğrula (ağsız, deterministik)
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_gamma_deviance,
    mean_poisson_deviance,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split

SEED = 42
KOK = Path(__file__).resolve().parent.parent

# Sabit, yorumlanabilir özellik kümesi (MTPL tarzı; sayısal + kategorik)
SAYISAL = ["DrivAge", "VehPower", "VehAge", "BonusMalus", "Density"]
KATEGORIK = ["VehBrand", "VehGas", "Area", "Region"]


# ------------------------------------------------------------------ veri
def _encode(df: pd.DataFrame) -> np.ndarray:
    """Sayısal + one-hot kategorik -> numpy matris (GLM girdisi)."""
    say = df[SAYISAL].to_numpy(dtype=float)
    kat = pd.get_dummies(df[KATEGORIK], drop_first=True).to_numpy(dtype=float)
    return np.hstack([say, kat])


def synthetic_data(n: int = 20000, seed: int = SEED) -> tuple[pd.DataFrame, str]:
    """SENTETİK demo verisi (açıkça etiketli). Gerçek veri DEĞİL.

    Veri üretim süreci (DGP) biliniyor: frekans ~ Poisson(rate), şiddet ~ Gamma.
    """
    rng = np.random.default_rng(seed)
    driv_age = rng.integers(18, 90, n).astype(float)
    veh_power = rng.integers(4, 15, n).astype(float)
    veh_age = rng.integers(0, 20, n).astype(float)
    bonus_malus = rng.integers(50, 150, n).astype(float)      # 100 = temiz
    density = rng.integers(1, 11, n).astype(float)
    veh_brand = rng.integers(1, 7, n).astype(str)
    veh_gas = rng.choice(["Regular", "Diesel"], n)
    area = rng.choice(["A", "B", "C", "D", "E", "F"], n)
    region = rng.choice([f"R{i}" for i in (11, 21, 22, 23, 24, 25, 26, 31, 41)], n)
    exposure = rng.uniform(0.5, 1.0, n)

    # --- frekans DGP: genç sürücü + yüksek malus + yoğun bölge -> daha çok hasar
    log_rate = (-3.0
                + 0.35 * (driv_age < 25).astype(float)
                - 0.010 * (driv_age - 40) / 10.0
                + 0.012 * (bonus_malus - 100) / 10.0
                + 0.020 * (density / 10.0))
    rate = np.exp(log_rate)
    claim_nb = rng.poisson(rate * exposure)

    # --- şiddet DGP: güçlü araç + eski araç + malus -> daha yüksek ortalama hasar
    log_sev = (7.5
               + 0.10 * (veh_power / 10.0)
               + 0.03 * (veh_age / 10.0)
               + 0.04 * (bonus_malus - 100) / 10.0)
    mean_sev = np.exp(log_sev)
    phi = 0.5  # Gamma dispersion
    avg_claim = np.where(
        claim_nb > 0,
        rng.gamma(shape=1.0 / phi, scale=mean_sev * phi),
        0.0,
    )

    df = pd.DataFrame({
        "DrivAge": driv_age, "VehPower": veh_power, "VehAge": veh_age,
        "BonusMalus": bonus_malus, "Density": density,
        "VehBrand": veh_brand, "VehGas": veh_gas, "Area": area, "Region": region,
        "Exposure": exposure, "ClaimNb": claim_nb, "AvgClaimAmount": avg_claim,
    })
    return df, f"sentetik-demo (seed={seed})"


def load_fremtpl2_freq(max_rows: int = 60000, timeout: int = 20) -> tuple[pd.DataFrame | None, str | None]:
    """freMTPL2freq (açık, CC0) — yalnız FREKANS bileşeni, en iyi çabayla indirme.

    Şiddet bileşeni freMTPL2'de ayrı veri setidir ve burada katılmaz; graybox'ın
    şiddet tarafı sentetik kalır (dürüst etiketle `load_data` içinde).
    """
    try:
        from sklearn.datasets import fetch_openml  # ağ + openml gerekir
        freq = fetch_openml("freMTPL2freq", version=1, as_frame=True, parser="auto")
        df = freq.sample(n=min(max_rows, len(freq)), random_state=SEED)
        keep = ["Area", "VehPower", "VehAge", "DrivAge", "BonusMalus",
                "VehBrand", "VehGas", "Density", "Region", "Exposure", "ClaimNb"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df["Exposure"] = df["Exposure"].astype(float)
        df["ClaimNb"] = df["ClaimNb"].astype(int)
        df["AvgClaimAmount"] = np.nan  # bu kaynakta şiddet yok
        return df, f"freMTPL2freq (CC0, openml 41214, n={len(df)})"
    except Exception:
        return None, None


def load_data(use_real: bool = False) -> tuple[pd.DataFrame, str]:
    """Graybox veri seti + dürüst kaynak etiketi.

    use_real=True -> freMTPL2 frekansı indirilebilirse onu, şiddet SENTETİK.
    use_real=False -> tamamen SENTETİK (deterministik).
    """
    if use_real:
        real_df, real_label = load_fremtpl2_freq()
        if real_df is not None:
            n = len(real_df)
            # şiddeti sentetik üret (açıkça etiketlenecek)
            sev_df, _ = synthetic_data(n=n, seed=SEED)
            real_df = real_df.reset_index(drop=True)
            real_df["AvgClaimAmount"] = np.where(
                real_df["ClaimNb"].to_numpy() > 0,
                sev_df["AvgClaimAmount"].to_numpy(),
                0.0,
            )
            label = f"{real_label} + sentetik şiddet (seed={SEED})"
            return real_df, label
    return synthetic_data(seed=SEED)


# ------------------------------------------------------------------ graybox
def _split(n: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """70 / 15 / 15 politika-bazlı bölme (random_state sabit, tekrarlanabilir)."""
    idx = np.arange(n)
    idx_tr, idx_te = train_test_split(idx, test_size=0.15, random_state=seed)
    idx_tr, idx_va = train_test_split(idx_tr, test_size=0.15 / 0.85, random_state=seed)
    return idx_tr, idx_va, idx_te


def _poisson_deviance(y_true, y_pred, sample_weight=None) -> float:
    return float(mean_poisson_deviance(y_true, y_pred, sample_weight=sample_weight))


def _fit_freq(X, y_freq, exposure, alpha) -> PoissonRegressor:
    m = PoissonRegressor(alpha=alpha, max_iter=1000)
    m.fit(X, y_freq, sample_weight=exposure)  # exposure -> ağırlık (offset)
    return m


def _fit_sev(X_sev, y_sev, alpha) -> GammaRegressor:
    m = GammaRegressor(alpha=alpha, max_iter=1000)
    m.fit(X_sev, y_sev)
    return m


def _alpha_araligi() -> tuple[float, float]:
    """log10(alpha) alt/üst sınır (alpha ~ 1e-6 .. 1000)."""
    return -6.0, 3.0


def ga_optimize(Xtr, yf_tr, w_tr, Xs_tr, ys_tr,
                Xva, yf_va, w_va, Xs_va, ys_va,
                nesil: int = 12, pop: int = 20, seed: int = SEED) -> dict:
    """Genetik Algoritma: iki GLM'in alpha hiperparametresini validation üzerinde arar.

    Kromozom = [log10(alpha_frekans), log10(alpha_şiddet)]. Amaç = toplam deviance
    (Poisson frekans + Gamma şiddet) minimizasyonu. GRID yok — mutasyon/çaprazlama.
    """
    rng = random.Random(seed)
    lo, hi = _alpha_araligi()

    _cache: dict = {}  # fitness memoizasyonu (her birey bir kez değerlendirilir)

    def hedef(krom: list[float]) -> float:
        key = (round(krom[0], 6), round(krom[1], 6))
        if key in _cache:
            return _cache[key]
        af, ase = 10.0 ** krom[0], 10.0 ** krom[1]
        mf = _fit_freq(Xtr, yf_tr, w_tr, af)
        pf_va = mf.predict(Xva) * w_va          # beklenen hasar sayısı
        fdev = _poisson_deviance(yf_va, pf_va)
        ms = _fit_sev(Xs_tr, ys_tr, ase)
        ps_va = ms.predict(Xs_va)
        gdev = float(mean_gamma_deviance(ys_va, ps_va))
        deger = fdev + gdev                      # küçültülecek
        _cache[key] = deger
        return deger

    def rastgele() -> list[float]:
        return [rng.uniform(lo, hi), rng.uniform(lo, hi)]

    pop_ = [rastgele() for _ in range(pop)]

    def turnuva(k=3):
        return min(rng.sample(pop_, k), key=hedef)[:]

    for _ in range(nesil):
        yeni = []
        # elitizm: en iyi 2 birey doğrudan geçer
        elit = sorted(pop_, key=hedef)[:2]
        yeni.extend([e[:] for e in elit])
        while len(yeni) < pop:
            a, b = turnuva(), turnuva()
            # çaprazlama (blend)
            cocuk = [rng.uniform(min(a[i], b[i]), max(a[i], b[i])) for i in range(2)]
            # mutasyon
            cocuk = [c + rng.gauss(0, 0.15) for c in cocuk]
            cocuk = [min(max(c, lo), hi) for c in cocuk]
            yeni.append(cocuk)
        pop_ = yeni

    en_iyi = min(pop_, key=hedef)
    return {
        "alpha_frekans": float(10.0 ** en_iyi[0]),
        "alpha_siddet": float(10.0 ** en_iyi[1]),
        "nesil": nesil, "populasyon": pop,
        "log10_alpha_frekans": float(en_iyi[0]),
        "log10_alpha_siddet": float(en_iyi[1]),
    }


def run(use_real: bool = False, seed: int = SEED) -> dict:
    """Uçtan uca graybox: veri -> split -> GA -> fit -> test metrikleri."""
    df, veri_etiket = load_data(use_real=use_real)
    n = len(df)
    X = _encode(df)
    y_freq = df["ClaimNb"].to_numpy(dtype=float)
    exposure = df["Exposure"].to_numpy(dtype=float)
    y_sev_all = df["AvgClaimAmount"].to_numpy(dtype=float)
    claim_mask = df["ClaimNb"].to_numpy() > 0

    idx_tr, idx_va, idx_te = _split(n, seed)

    def parcalar(idx):
        cm = claim_mask[idx]
        return (X[idx], y_freq[idx], exposure[idx],
                X[idx][cm], y_sev_all[idx][cm])

    Xtr, yf_tr, w_tr, Xs_tr, ys_tr = parcalar(idx_tr)
    Xva, yf_va, w_va, Xs_va, ys_va = parcalar(idx_va)
    Xte, yf_te, w_te, Xs_te, ys_te = parcalar(idx_te)

    ga = ga_optimize(Xtr, yf_tr, w_tr, Xs_tr, ys_tr,
                     Xva, yf_va, w_va, Xs_va, ys_va, seed=seed)

    # final fit (en iyi alpha ile), test üzerinde değerlendir
    mf = _fit_freq(Xtr, yf_tr, w_tr, ga["alpha_frekans"])
    ms = _fit_sev(Xs_tr, ys_tr, ga["alpha_siddet"])

    # frekans: beklenen hasar sayısı = rate * exposure
    freq_pred_te = mf.predict(Xte) * w_te
    fdev_te = _poisson_deviance(yf_te, freq_pred_te)

    # şiddet: E[hasar | hasar var]
    sev_pred_te = ms.predict(Xs_te)
    sev_mse = float(mean_squared_error(ys_te, sev_pred_te))
    sev_mae = float(mean_absolute_error(ys_te, sev_pred_te))

    # saf prim (tüm politika): rate * E[şiddet] ; gerçekleşen = ClaimNb*Avg / exposure
    rate_te = mf.predict(Xte)                       # beklenen frekans oranı (yıl başına)
    sev_pred_all = ms.predict(Xte)                  # tüm politika için beklenen şiddet
    prim_pred = rate_te * sev_pred_all
    prim_gercek = yf_te * y_sev_all[idx_te] / w_te
    prim_mse = float(mean_squared_error(prim_gercek, prim_pred))
    prim_mae = float(mean_absolute_error(prim_gercek, prim_pred))

    return {
        "veri": veri_etiket,
        "veri_satir": n,
        "split": {"train_oran": 0.7, "val_oran": 0.15, "test_oran": 0.15,
                  "n_train": int(len(idx_tr)), "n_val": int(len(idx_va)),
                  "n_test": int(len(idx_te))},
        "ga": ga,
        "metrikler": {
            "frekans_poisson_deviance_test": round(fdev_te, 6),
            "siddet_mse_test": round(sev_mse, 4),
            "siddet_mae_test": round(sev_mae, 4),
            "saf_prim_mse_test": round(prim_mse, 4),
            "saf_prim_mae_test": round(prim_mae, 4),
        },
        "saf_prim_ort_test": round(float(np.mean(prim_pred)), 4),
        "hasar_sikligi_test": round(float(np.mean(yf_te / w_te)), 4),
    }


# ------------------------------------------------------------------ test
def test() -> int:
    """Ağsız, deterministik kendini doğrulama."""
    kontrols = []

    def kontrol(ad, kosul, detay=""):
        kontrols.append((ad, bool(kosul), detay))

    r = run(use_real=False, seed=SEED)

    # 1) split oranları (70/15/15)
    n = r["veri_satir"]
    s = r["split"]
    kontrol("split: train ~0.70", abs(s["n_train"] / n - 0.70) < 0.02,
            f"{s['n_train']}/{n}")
    kontrol("split: val ~0.15", abs(s["n_val"] / n - 0.15) < 0.02,
            f"{s['n_val']}/{n}")
    kontrol("split: test ~0.15", abs(s["n_test"] / n - 0.15) < 0.02,
            f"{s['n_test']}/{n}")

    # 2) GA parametre buldu (pozitif, sonlu)
    ga = r["ga"]
    kontrol("GA: alpha_frekans pozitif", ga["alpha_frekans"] > 0,
            f"{ga['alpha_frekans']:.6f}")
    kontrol("GA: alpha_siddet pozitif", ga["alpha_siddet"] > 0,
            f"{ga['alpha_siddet']:.6f}")
    kontrol("GA: iki alpha da sonlu", math.isfinite(ga["alpha_frekans"])
            and math.isfinite(ga["alpha_siddet"]))

    # 3) metrikler hesaplandı
    m = r["metrikler"]
    kontrol("metrik: poisson deviance var", "frekans_poisson_deviance_test" in m
            and m["frekans_poisson_deviance_test"] >= 0)
    kontrol("metrik: MSE/MAE sonlu", all(math.isfinite(v) for v in m.values()))

    # 4) veri etiketi dürüst
    kontrol("veri: sentetik açıkça etiketli",
            "sentetik" in r["veri"].lower() and "seed" in r["veri"].lower(),
            r["veri"])

    ok = sum(1 for _, k, _ in kontrols if k)
    kal = len(kontrols) - ok
    for ad, k, detay in kontrols:
        isaret = "[OK]    " if k else "[KALDI] "
        print(f"  {isaret} {ad}" + (f"  {detay}" if detay else ""))
    print(f"SONUC: {ok} başarılı, {kal} KALDI · veri={r['veri']} · "
          f"alpha=(f:{r['ga']['alpha_frekans']:.5f}, s:{r['ga']['alpha_siddet']:.5f})")
    return 0 if kal == 0 else 1


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--test" in args:
        return test()
    use_real = "--real" in args
    r = run(use_real=use_real)
    print(json_dumps(r))
    return 0


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
